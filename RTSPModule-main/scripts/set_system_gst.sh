export GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

# Add WSL2 library path for NVIDIA drivers (libcuda, libnvcuvid, libnvidia-encode)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/usr/bin:$PATH

# Clear any conflicting custom paths
unset GST_PLUGIN_SYSTEM_PATH
