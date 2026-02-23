/// Sink element module — public API and registration.

mod imp;

use gst::glib;
use gst::prelude::*;

glib::wrapper! {
    pub struct WebRtcRsSink(ObjectSubclass<imp::WebRtcRsSink>) @extends gst_base::BaseSink, gst::Element, gst::Object;
}

/// Register the element with the GStreamer plugin.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "webrtcrs_sink",
        gst::Rank::None,
        WebRtcRsSink::static_type(),
    )
}
