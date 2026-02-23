/// WebRTC Manager — owns the webrtc-rs runtime, tracks, and peer connections.
///
/// Design:
/// - One `TrackLocalStaticRTP` per manager (per stream).
/// - Multiple `RTCPeerConnection`s can share the same track via `Arc`.
/// - `write_rtp()` writes packets to the track — webrtc-rs fans out internally.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use tokio::sync::Mutex;
use uuid::Uuid;

use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_H264};
use webrtc::api::APIBuilder;
use webrtc::ice_transport::ice_connection_state::RTCIceConnectionState;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::rtp_transceiver::rtp_codec::{RTCRtpCodecCapability, RTCRtpCodecParameters, RTPCodecType};
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;
use webrtc::track::track_local::{TrackLocal, TrackLocalWriter};

/// Information about a connected peer.
struct PeerInfo {
    _peer_connection: Arc<RTCPeerConnection>,
}

/// Default maximum number of concurrent WebRTC peers per stream.
const DEFAULT_MAX_PEERS: usize = 10;

/// Manages WebRTC tracks and peer connections for a single stream.
pub struct WebRtcManager {
    stream_id: String,
    /// The shared video track — all peers receive data from this single track.
    video_track: Arc<TrackLocalStaticRTP>,
    /// The webrtc-rs API instance (holds media engine config).
    api: webrtc::api::API,
    /// Connected peers, keyed by UUID.
    /// Wrapped in `Arc` so ICE callbacks can hold a reference for auto-cleanup.
    peers: Arc<Mutex<HashMap<String, PeerInfo>>>,
    /// Maximum allowed concurrent peers (prevents unbounded resource growth).
    max_peers: usize,
}

impl WebRtcManager {
    /// Create a new manager for the given stream ID.
    pub async fn new(stream_id: &str) -> Result<Self> {
        // Sanitize stream_id for SDP MSID (RFC 8830 §2: token chars only, no whitespace).
        // Camera names like "Camera 3" would produce a malformed `a=msid:Camera 3 video-0` line.
        let sdp_stream_id: String = stream_id
            .chars()
            .map(|c| if c.is_ascii_whitespace() { '_' } else { c })
            .collect();

        // Configure the media engine with H.264 ONLY.
        //
        // Using register_default_codecs() registers VP8/VP9/AV1/H.265 as well.
        // Browsers prefer VP8/VP9, so the SDP negotiation resolves to VP8/VP9 —
        // but our GStreamer pipeline sends raw H.264 RTP (rtph264pay), causing the
        // browser to silently discard every packet.
        //
        // By registering only H.264, the SDP answer always says "send me H.264"
        // regardless of the browser's codec preference ordering.
        let mut media_engine = MediaEngine::default();
        media_engine.register_codec(
            RTCRtpCodecParameters {
                capability: RTCRtpCodecCapability {
                    mime_type: MIME_TYPE_H264.to_owned(),
                    clock_rate: 90000,
                    channels: 0,
                    sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f"
                        .to_owned(),
                    rtcp_feedback: vec![],
                },
                payload_type: 96,
                ..Default::default()
            },
            RTPCodecType::Video,
        )?;

        // Create interceptor registry (handles RTCP feedback, etc.)
        let mut registry = Registry::new();
        registry = register_default_interceptors(registry, &mut media_engine)?;

        let api = APIBuilder::new()
            .with_media_engine(media_engine)
            .with_interceptor_registry(registry)
            .build();

        // Create the shared video track — use the sanitized ID in the SDP
        let video_track = Arc::new(TrackLocalStaticRTP::new(
            RTCRtpCodecCapability {
                mime_type: MIME_TYPE_H264.to_owned(),
                clock_rate: 90000,
                sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f"
                    .to_owned(),
                ..Default::default()
            },
            "video-0".to_owned(),
            sdp_stream_id,  // RFC 8830-compliant: no whitespace
        ));

        Ok(Self {
            stream_id: stream_id.to_owned(),  // Keep original for /streams endpoint
            video_track,
            api,
            peers: Arc::new(Mutex::new(HashMap::new())),
            max_peers: DEFAULT_MAX_PEERS,
        })
    }

    /// Get the stream ID.
    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    /// Get the current peer count.
    pub async fn peer_count(&self) -> usize {
        self.peers.lock().await.len()
    }

    /// Add a new peer connection from a browser SDP offer.
    /// Returns `(peer_id, sdp_answer)`.
    pub async fn add_peer(&self, offer: RTCSessionDescription) -> Result<(String, RTCSessionDescription)> {
        // P1: Reject if we've hit the per-stream peer limit.
        {
            let peers = self.peers.lock().await;
            if peers.len() >= self.max_peers {
                bail!("Max peers ({}) reached for stream '{}'", self.max_peers, self.stream_id);
            }
        }

        let peer_id = Uuid::new_v4().to_string();

        let config = RTCConfiguration {
            ice_servers: vec![RTCIceServer {
                urls: vec!["stun:stun.l.google.com:19302".to_owned()],
                ..Default::default()
            }],
            ..Default::default()
        };

        // Create a new peer connection
        let peer_connection = Arc::new(self.api.new_peer_connection(config).await?);

        // Add the shared video track to this peer
        let rtp_sender = peer_connection
            .add_track(self.video_track.clone() as Arc<dyn TrackLocal + Send + Sync>)
            .await?;

        // Spawn a task to read and discard RTCP packets (required by webrtc-rs)
        tokio::spawn(async move {
            let mut rtcp_buf = vec![0u8; 1500];
            while let Ok((_, _)) = rtp_sender.read(&mut rtcp_buf).await {}
        });

        // Monitor ICE connection state — auto-remove disconnected peers
        let pid = peer_id.clone();
        let peers_ref = self.peers.clone();
        let pc_ref = peer_connection.clone();
        peer_connection.on_ice_connection_state_change(Box::new(move |state| {
            let pid = pid.clone();
            let peers_ref = peers_ref.clone();
            let pc_ref = pc_ref.clone();
            Box::pin(async move {
                eprintln!(
                    "[webrtcrs] Peer {}: ICE state -> {:?}",
                    &pid[..8],
                    state
                );
                if state == RTCIceConnectionState::Disconnected
                    || state == RTCIceConnectionState::Failed
                    || state == RTCIceConnectionState::Closed
                {
                    // P1: Only close if we actually owned this peer (prevents double-close
                    // when remove_peer() and ICE callback race).
                    let removed = {
                        let mut peers = peers_ref.lock().await;
                        let was_present = peers.remove(&pid).is_some();
                        if was_present {
                            eprintln!(
                                "[webrtcrs] Peer {} removed (remaining: {})",
                                &pid[..8],
                                peers.len()
                            );
                        }
                        was_present
                    };
                    if removed {
                        if let Err(e) = pc_ref.close().await {
                            eprintln!("[webrtcrs] Error closing peer {}: {}", &pid[..8], e);
                        }
                    }
                }
            })
        }));

        // P0 fix: Insert peer BEFORE set_remote_description so the ICE callback
        // can find it if ICE transitions happen immediately during negotiation.
        let info = PeerInfo {
            _peer_connection: peer_connection.clone(),
        };
        self.peers.lock().await.insert(peer_id.clone(), info);

        // Set the remote description (browser's offer)
        peer_connection.set_remote_description(offer).await?;

        // Create an answer
        let answer = peer_connection.create_answer(None).await?;

        // Set the local description (starts ICE gathering)
        let mut gather_complete = peer_connection.gathering_complete_promise().await;
        peer_connection.set_local_description(answer).await?;

        // P0 fix: Wait for ICE gathering with a timeout to prevent indefinite hang
        // when STUN is unreachable.
        match tokio::time::timeout(Duration::from_secs(10), gather_complete.recv()).await {
            Ok(_) => {},
            Err(_) => {
                eprintln!("[webrtcrs] Peer {}: ICE gathering timed out (10s)", &peer_id[..8]);
                // Remove the peer we inserted and clean up
                self.peers.lock().await.remove(&peer_id);
                let _ = peer_connection.close().await;
                bail!("ICE gathering timed out after 10 seconds");
            }
        }

        let local_desc = peer_connection
            .local_description()
            .await
            .context("Failed to get local description after ICE gathering")?;

        eprintln!(
            "[webrtcrs] Peer {} added (total: {})",
            &peer_id[..8],
            self.peers.lock().await.len()
        );

        Ok((peer_id, local_desc))
    }

    /// Remove a peer connection.
    pub async fn remove_peer(&self, peer_id: &str) -> Result<()> {
        let mut peers = self.peers.lock().await;
        if let Some(info) = peers.remove(peer_id) {
            info._peer_connection.close().await?;
            eprintln!(
                "[webrtcrs] Peer {} removed (remaining: {})",
                &peer_id[..8],
                peers.len()
            );
        }
        Ok(())
    }

    /// Write an RTP packet to the shared track.
    /// This is called from the GStreamer render thread (synchronous context).
    ///
    /// P0 fix: Uses `spawn` (fire-and-forget) instead of `block_on` to avoid
    /// deadlocking the GStreamer streaming thread when the Tokio thread pool
    /// is saturated. RTP is inherently lossy, so dropped writes are acceptable.
    pub fn write_rtp(&self, rtp_data: &[u8], rt_handle: &tokio::runtime::Handle) {
        let data = bytes::Bytes::copy_from_slice(rtp_data);
        let track = self.video_track.clone();

        // Fire-and-forget: never block the GStreamer streaming thread.
        rt_handle.spawn(async move {
            if let Err(e) = track.write(&data).await {
                // Only log occasionally to avoid spam
                static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count % 1000 == 0 {
                    eprintln!("[webrtcrs] RTP write error (count={}): {}", count, e);
                }
            }
        });
    }

    /// Gracefully close all peer connections.
    pub async fn shutdown(&self) {
        let mut peers = self.peers.lock().await;
        for (id, info) in peers.drain() {
            let short_id = if id.len() >= 8 { &id[..8] } else { &id };
            if let Err(e) = info._peer_connection.close().await {
                eprintln!("[webrtcrs] Error closing peer {short_id}: {e}");
            }
        }
        eprintln!("[webrtcrs] All peers closed.");
    }
}
