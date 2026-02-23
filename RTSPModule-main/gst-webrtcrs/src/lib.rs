/// GStreamer WebRTC-RS Sink Plugin
///
/// Registers the `webrtcrs_sink` element with GStreamer.
/// This element accepts RTP packets and streams them to browsers via WebRTC,
/// using the pure-Rust `webrtc-rs` implementation.

mod hub;
mod signaling;
mod sink;
mod webrtc_manager;

use gst::glib;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    sink::register(plugin)?;
    Ok(())
}

gst::plugin_define!(
    webrtcrs,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "MIT",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
