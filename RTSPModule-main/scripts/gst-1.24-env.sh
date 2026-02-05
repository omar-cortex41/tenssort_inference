export GST_PREFIX=/opt/gstreamer-1.24/install
export PATH=$GST_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$GST_PREFIX/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=$GST_PREFIX/lib/x86_64-linux-gnu/gstreamer-1.0
export PKG_CONFIG_PATH=$GST_PREFIX/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

echo "GStreamer 1.24 environment activated"
echo "  GST_PREFIX: $GST_PREFIX"
gst-inspect-1.0 --version
