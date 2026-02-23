/// Process-wide WebRTC Hub
///
/// Maintains a shared registry of `WebRtcManager` instances (one per stream) and
/// runs a **single** Axum signaling server for all of them.
///
/// Lifecycle:
/// - First `register()` call starts the Axum server.
/// - Subsequent `register()` calls just insert into the map.
/// - `unregister()` removes an entry; when the last entry is removed the server
///   is shut down so the port is freed.

use std::sync::Arc;

use once_cell::sync::Lazy;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::signaling;
use crate::webrtc_manager::WebRtcManager;

// ─── Global hub instance ──────────────────────────────────────────────────────

static HUB: Lazy<std::sync::Mutex<Hub>> = Lazy::new(|| std::sync::Mutex::new(Hub::new()));

/// Access the process-wide hub (sync lock — only held briefly for bookkeeping).
pub fn global() -> std::sync::MutexGuard<'static, Hub> {
    HUB.lock().expect("WebRTC hub mutex poisoned")
}

// ─── Hub ─────────────────────────────────────────────────────────────────────

/// The signaling server's shared application state.
/// Using a Vec of tuples rather than a HashMap so iteration always returns
/// streams in insertion order (= camera index order from start_streaming calls).
pub type SharedManagers = Arc<RwLock<Vec<(String, Arc<WebRtcManager>)>>>;

pub struct Hub {
    /// Registered managers in insertion order (cam_index order).
    managers: SharedManagers,
    /// Handle to the running Axum server task (None until first register).
    server_handle: Option<JoinHandle<()>>,
    /// Token to cancel the server on shutdown.
    cancel_token: CancellationToken,
    /// The port the server is (or will be) listening on.
    port: u16,
    /// Tokio runtime used by the server (kept alive until last unregister).
    _runtime: Option<tokio::runtime::Runtime>,
}

impl Hub {
    fn new() -> Self {
        Self {
            managers: Arc::new(RwLock::new(Vec::new())),
            server_handle: None,
            cancel_token: CancellationToken::new(),
            port: 9000,
            _runtime: None,
        }
    }

    /// Register a stream with the hub.
    ///
    /// If this is the first stream, the Axum server is started on `port`.
    /// If a server is already running on a **different** port, the call is
    /// rejected (all sinks must agree on the same port).
    ///
    /// # Panics
    /// Panics if called with a different port while the server is already running.
    pub fn register(&mut self, stream_id: String, manager: Arc<WebRtcManager>, port: u16) {
        if self.server_handle.is_some() {
            // Server already running — just add the manager.
            if self.port != port {
                eprintln!(
                    "[webrtcrs-hub] WARNING: sink requested port {} but server already running on {}. \
                     Using existing port.",
                    port, self.port
                );
            }
        } else {
            // First registration — start the server.
            self.port = port;
            self.cancel_token = CancellationToken::new();

            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(2)
                .thread_name("webrtcrs-hub")
                .build()
                .expect("Failed to create Tokio runtime for WebRTC hub");

            let managers_clone  = self.managers.clone();
            let cancel          = self.cancel_token.clone();

            let handle = runtime.spawn(async move {
                if let Err(e) = signaling::run_server(managers_clone, port, cancel).await {
                    eprintln!("[webrtcrs-hub] Signaling server error: {e}");
                }
            });

            self.server_handle = Some(handle);
            self._runtime = Some(runtime);

            eprintln!("[webrtcrs-hub] Signaling server started on port {port}");
        }

        // Insert the manager in registration order (Vec push).
        if let Some(rt) = self._runtime.as_ref() {
            rt.block_on(async {
                let mut list = self.managers.write().await;
                // Avoid duplicates (idempotent re-register)
                if !list.iter().any(|(id, _)| id == &stream_id) {
                    list.push((stream_id.clone(), manager));
                }
            });
        }

        eprintln!("[webrtcrs-hub] Stream '{}' registered (total: {})",
            stream_id,
            self._runtime.as_ref()
                .map(|rt| rt.block_on(async { self.managers.read().await.len() }))
                .unwrap_or(0)
        );
    }

    /// Unregister a stream from the hub.
    ///
    /// When the last stream is removed the signaling server is shut down and
    /// the Tokio runtime is dropped.
    pub fn unregister(&mut self, stream_id: &str) {
        let remaining = if let Some(rt) = self._runtime.as_ref() {
            rt.block_on(async {
                let mut list = self.managers.write().await;
                list.retain(|(id, _)| id != stream_id);
                list.len()
            })
        } else {
            0
        };

        eprintln!("[webrtcrs-hub] Stream '{stream_id}' unregistered (remaining: {remaining})");

        if remaining == 0 {
            // Cancel the server and tear down the runtime.
            self.cancel_token.cancel();

            if let Some(handle) = self.server_handle.take() {
                if let Some(rt) = self._runtime.as_ref() {
                    // Give the server a moment to finish graceful shutdown.
                    let _ = rt.block_on(async {
                        tokio::time::timeout(
                            std::time::Duration::from_secs(3),
                            handle,
                        ).await
                    });
                }
            }

            // Drop runtime — releases all hub tasks.
            self._runtime = None;
            eprintln!("[webrtcrs-hub] Signaling server stopped (no more streams).");
        }
    }
}
