FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV GST_VERSION=1.24.0


RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    ninja-build \
    bison \
    flex \
    libglib2.0-dev \
    libssl-dev \
    cmake \
    pkg-config \
    libmount-dev \
    libz-dev \
    zlib1g-dev \
    libyaml-cpp-dev \
    libgl1-mesa-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libva-dev \
    libdrm-dev \
    libnice-dev \
    libsrtp2-dev \
    libusrsctp-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install meson scikit-build-core pybind11 build



WORKDIR /working/gst-build


RUN git clone --branch $GST_VERSION --depth 1 https://gitlab.freedesktop.org/gstreamer/gstreamer.git

WORKDIR /working/gst-build/gstreamer
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
ENV CPATH=/usr/local/cuda/include:$CPATH
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH

RUN meson setup builddir \
    --prefix=/opt/gst \
    -Dgst-plugins-bad:nvcodec=enabled \
    -Dgst-plugins-bad:va=enabled \
    -Dgst-plugins-bad:webrtc=enabled \
    -Dlibav=enabled \
    -Dtests=disabled \
    -Dexamples=disabled \
    -Ddoc=disabled \
    -Dpython=disabled


RUN ninja -C builddir install


WORKDIR /working/app
COPY . .


ENV PKG_CONFIG_PATH=/opt/gst/lib/x86_64-linux-gnu/pkgconfig
ENV LD_LIBRARY_PATH=/opt/gst/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Step 1: Build the native module with cmake
RUN rm -rf build && mkdir build && cd build && \
    cmake .. -DCMAKE_PREFIX_PATH=/opt/gst && \
    make -j$(nproc)

# Step 2: Build wheel using pre-compiled _core.so (no rebuild)
RUN pip3 wheel --no-build-isolation --no-deps -w /working/wheels .


FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libssl3 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libyaml-cpp-dev \
    libgl1 \
    libegl1 \
    cuda-nvrtc-12-8 \
    libva-drm2 \
    libnice10 \
    libsrtp2-1 \
    libusrsctp2 \
    vainfo \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12 /usr/lib/x86_64-linux-gnu/libnvrtc.so


# Copy GStreamer and built wheel
COPY --from=builder /opt/gst /opt/gst
COPY --from=builder /working/wheels /wheels

ENV PATH=/opt/gst/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/gst/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=/opt/gst/lib/x86_64-linux-gnu/pkgconfig

# Install rtspmodule from wheel + runtime dependencies
RUN pip3 install --no-cache-dir /wheels/*.whl \
    cupy-cuda12x \
    opencv-python \
    pyyaml \
    pynvml \
    psutil \
    aiohttp \
    simplejpeg \
    && rm -rf /wheels

WORKDIR /workspace

CMD ["/bin/bash"]
