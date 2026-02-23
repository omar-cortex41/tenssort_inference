/// GstBaseSink implementation for `webrtcrs_sink`.
///
/// Receives RTP-packetized H.264 buffers from upstream and forwards them
/// to all connected WebRTC peers via the `WebRtcManager`.
///
/// Multiple instances of this element share a single HTTP signaling server
/// via the process-wide `hub::Hub` singleton.  All sinks must use the same
/// `signaling-port`; streams are distinguished by their `stream-id` property.

use std::sync::{Arc, Mutex};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;

use once_cell::sync::Lazy;

use crate::hub;
use crate::webrtc_manager::WebRtcManager;

/// Default signaling port — all sinks share this one port.
const DEFAULT_SIGNALING_PORT: u32 = 9000;
const DEFAULT_STREAM_ID: &str = "default";

/// Properties exposed to GStreamer.
#[derive(Debug, Clone)]
struct Settings {
    stream_id: String,
    signaling_port: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            stream_id: DEFAULT_STREAM_ID.to_owned(),
            signaling_port: DEFAULT_SIGNALING_PORT,
        }
    }
}

/// Internal state managed by the element.
struct State {
    manager: Arc<WebRtcManager>,
    /// Tokio runtime for RTP write bridging in render().
    rt_handle: tokio::runtime::Handle,
    /// Tokio runtime (kept alive for the element's lifetime).
    _runtime: tokio::runtime::Runtime,
    /// The stream_id registered with the hub (needed for unregister on stop).
    stream_id: String,
}

/// The ObjectSubclass implementation for the sink element.
pub struct WebRtcRsSink {
    settings: Mutex<Settings>,
    state: Mutex<Option<State>>,
}

impl Default for WebRtcRsSink {
    fn default() -> Self {
        Self {
            settings: Mutex::new(Settings::default()),
            state: Mutex::new(None),
        }
    }
}

/// Property IDs.
static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
    vec![
        glib::ParamSpecString::builder("stream-id")
            .nick("Stream ID")
            .blurb("Unique identifier for this stream (used in signaling URLs)")
            .default_value(Some(DEFAULT_STREAM_ID))
            .build(),
        glib::ParamSpecUInt::builder("signaling-port")
            .nick("Signaling Port")
            .blurb("Shared HTTP port for the WebRTC signaling server (one server for all streams)")
            .minimum(1)
            .maximum(65535)
            .default_value(DEFAULT_SIGNALING_PORT)
            .build(),
    ]
});

static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
    // Accept RTP-packetized H.264 (from both H.264 pass-through and H.265 transcode paths)
    let caps = gst::Caps::builder("application/x-rtp")
        .field("media", "video")
        .build();

    vec![gst::PadTemplate::new(
        "sink",
        gst::PadDirection::Sink,
        gst::PadPresence::Always,
        &caps,
    )
    .unwrap()]
});

#[glib::object_subclass]
impl ObjectSubclass for WebRtcRsSink {
    const NAME: &'static str = "GstWebRtcRsSink";
    type Type = super::WebRtcRsSink;
    type ParentType = gst_base::BaseSink;
}

impl ObjectImpl for WebRtcRsSink {
    fn properties() -> &'static [glib::ParamSpec] {
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut settings = self.settings.lock().unwrap();
        match pspec.name() {
            "stream-id" => {
                settings.stream_id = value.get::<String>().expect("type checked upstream");
            }
            "signaling-port" => {
                settings.signaling_port = value.get::<u32>().expect("type checked upstream");
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings.lock().unwrap();
        match pspec.name() {
            "stream-id" => settings.stream_id.to_value(),
            "signaling-port" => settings.signaling_port.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for WebRtcRsSink {}

impl ElementImpl for WebRtcRsSink {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "WebRTC-RS Sink",
                "Sink/Network/WebRTC",
                "Streams RTP video to browsers via WebRTC (shared single-port signaling)",
                "RTSPModule Team",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        PAD_TEMPLATES.as_ref()
    }
}

impl BaseSinkImpl for WebRtcRsSink {
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let settings = self.settings.lock().unwrap().clone();

        gst::info!(
            gst::CAT_DEFAULT,
            "Starting WebRTC-RS sink: stream_id={}, port={}",
            settings.stream_id,
            settings.signaling_port
        );

        // Build a dedicated Tokio runtime for async bridging in render().
        // The hub has its own runtime; this one is just for write_rtp().
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(2)
            .thread_name("webrtcrs-rtp")
            .build()
            .map_err(|e| {
                gst::error_msg!(
                    gst::ResourceError::Failed,
                    ["Failed to create Tokio runtime: {}", e]
                )
            })?;

        // Create the WebRTC manager for this stream.
        let manager = runtime.block_on(async {
            WebRtcManager::new(&settings.stream_id).await
        }).map_err(|e| {
            gst::error_msg!(
                gst::ResourceError::Failed,
                ["Failed to create WebRTC manager: {}", e]
            )
        })?;

        let manager = Arc::new(manager);
        let rt_handle = runtime.handle().clone();
        let stream_id = settings.stream_id.clone();
        let port = settings.signaling_port as u16;

        // Register with the global hub — this starts the HTTP server if it isn't
        // already running, or simply adds this stream to the existing server's map.
        hub::global().register(stream_id.clone(), manager.clone(), port);

        let mut state = self.state.lock().unwrap();
        *state = Some(State {
            manager,
            rt_handle,
            _runtime: runtime,
            stream_id,
        });

        eprintln!(
            "[webrtcrs_sink] Started — stream_id='{}', signaling on port {}",
            settings.stream_id, settings.signaling_port
        );

        Ok(())
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        let mut state_guard = self.state.lock().unwrap();
        if let Some(state) = state_guard.take() {
            eprintln!("[webrtcrs_sink] Stopping stream '{}'...", state.stream_id);

            // Close all WebRTC peer connections for this stream.
            state.rt_handle.block_on(state.manager.shutdown());

            // Unregister from the hub — shuts down the HTTP server when last
            // stream is removed.
            hub::global().unregister(&state.stream_id);

            // Runtime drops here — all tasks already finished.
            drop(state);
            eprintln!("[webrtcrs_sink] Stopped.");
        }
        Ok(())
    }

    fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
        let state_guard = self.state.lock().unwrap();
        let state = state_guard.as_ref().ok_or_else(|| {
            gst::element_error!(
                self.obj(),
                gst::StreamError::Failed,
                ["Element not started"]
            );
            gst::FlowError::Error
        })?;

        // Map the GstBuffer to read RTP packet bytes
        let map = buffer.map_readable().map_err(|_| {
            gst::element_error!(
                self.obj(),
                gst::StreamError::Failed,
                ["Failed to map buffer"]
            );
            gst::FlowError::Error
        })?;

        let rtp_data = map.as_slice();

        // Forward the RTP packet to all connected WebRTC peers.
        // Uses the stored runtime handle to bridge sync→async.
        state.manager.write_rtp(rtp_data, &state.rt_handle);

        Ok(gst::FlowSuccess::Ok)
    }
}
