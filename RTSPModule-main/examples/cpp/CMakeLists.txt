# =============================================================================
# Example CMakeLists.txt for using libRTSPCore.a static library
# =============================================================================
#
# This CMake configuration demonstrates how to link against the pre-built
# RTSPCore library to create standalone C++ applications.
#
# Prerequisites:
#   1. Build the main RTSPModule project first to generate libRTSPCore.a
#   2. Ensure GStreamer and CUDA are available on the system
#
# Build Instructions:
#   mkdir build && cd build
#   cmake .. -DRTSPCORE_LIB_PATH=/path/to/libRTSPCore.a
#   make
#
# Run:
#   ./rtsp_example /path/to/config.yaml
#
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(RTSPCoreExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Configurable Paths
# =============================================================================

# Path to the pre-built RTSPCore library (can be overridden via -D)
set(RTSPCORE_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../build/libRTSPCore.a"
    CACHE FILEPATH "Path to libRTSPCore.a")

# Path to RTSPCore include directory
set(RTSPCORE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    CACHE PATH "Path to RTSPCore include directory")

# Verify the library exists
if(NOT EXISTS "${RTSPCORE_LIB_PATH}")
    message(FATAL_ERROR 
        "libRTSPCore.a not found at: ${RTSPCORE_LIB_PATH}\n"
        "Please build the main RTSPModule project first, or specify the correct path:\n"
        "  cmake .. -DRTSPCORE_LIB_PATH=/path/to/libRTSPCore.a"
    )
endif()

message(STATUS "Using RTSPCore library: ${RTSPCORE_LIB_PATH}")
message(STATUS "Using RTSPCore includes: ${RTSPCORE_INCLUDE_DIR}")

# =============================================================================
# Dependencies
# =============================================================================

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(yaml-cpp REQUIRED)

# GStreamer packages
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-video-1.0
    gstreamer-app-1.0
    gstreamer-sdp-1.0
)

# Optional: gstreamer-cuda
pkg_check_modules(GST_CUDA gstreamer-cuda-1.0)
if(GST_CUDA_FOUND)
    message(STATUS "Found gstreamer-cuda-1.0 - GPU acceleration available")
    list(APPEND GST_INCLUDE_DIRS ${GST_CUDA_INCLUDE_DIRS})
    list(APPEND GST_LIBRARIES ${GST_CUDA_LIBRARIES})
    list(APPEND GST_LIBRARY_DIRS ${GST_CUDA_LIBRARY_DIRS})
    add_definitions(-DHAVE_GST_CUDA)
endif()

link_directories(${GST_LIBRARY_DIRS})

# =============================================================================
# Example Executable
# =============================================================================

add_executable(rtsp_example
    main.cpp
)

target_include_directories(rtsp_example PRIVATE
    ${RTSPCORE_INCLUDE_DIR}
    ${GST_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link order matters: static library first, then its dependencies
target_link_libraries(rtsp_example PRIVATE
    ${RTSPCORE_LIB_PATH}
    ${GST_LIBRARIES}
    yaml-cpp::yaml-cpp
    CUDA::cudart
    pthread
)

# Optional DeepStream support
set(DEEPSTREAM_PATHS
    "/opt/nvidia/deepstream/deepstream"
    "/opt/nvidia/deepstream/deepstream-7.0"
    "/opt/nvidia/deepstream/deepstream-7.1"
    "/opt/nvidia/deepstream/deepstream-8.0"
)

foreach(DS_PATH ${DEEPSTREAM_PATHS})
    if(EXISTS "${DS_PATH}/lib/libnvbufsurface.so")
        message(STATUS "DeepStream SDK found at ${DS_PATH}")
        target_compile_definitions(rtsp_example PRIVATE HAVE_DEEPSTREAM)
        target_include_directories(rtsp_example PRIVATE ${DS_PATH}/sources/includes)
        find_library(NVBUFSURFACE nvbufsurface PATHS ${DS_PATH}/lib NO_DEFAULT_PATH)
        if(NVBUFSURFACE)
            target_link_libraries(rtsp_example PRIVATE ${NVBUFSURFACE})
        endif()
        break()
    endif()
endforeach()

# =============================================================================
# Installation (optional)
# =============================================================================

install(TARGETS rtsp_example RUNTIME DESTINATION bin)
