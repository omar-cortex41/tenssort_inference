cmake_minimum_required(VERSION 3.18)
project(RTSPModule LANGUAGES CXX CUDA)

# Add cmake/modules to module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(pybind11)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)

# Core GStreamer packages (always required)
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-video-1.0
    gstreamer-app-1.0
    gstreamer-sdp-1.0
    gstreamer-allocators-1.0
)

# gstreamer-cuda-1.0 is optional
pkg_check_modules(GST_CUDA gstreamer-cuda-1.0)
if(GST_CUDA_FOUND)
    message(STATUS "Found gstreamer-cuda-1.0 - enabling GstCudaMemory support")
    list(APPEND GST_INCLUDE_DIRS ${GST_CUDA_INCLUDE_DIRS})
    list(APPEND GST_LIBRARIES ${GST_CUDA_LIBRARIES})
    list(APPEND GST_LIBRARY_DIRS ${GST_CUDA_LIBRARY_DIRS})
    add_definitions(-DHAVE_GST_CUDA)
else()
    message(STATUS "gstreamer-cuda-1.0 not found - will use DeepStream NVMM or CPU fallback")
endif()

# Add GStreamer library directories to linker search path
# Required when GStreamer is installed in a non-standard location (e.g., /opt/gst)
link_directories(${GST_LIBRARY_DIRS})

find_package(yaml-cpp REQUIRED)

# Core RTSP Library
add_library(RTSPCore STATIC
    src/core/rtsp_client.cpp
    src/core/stream_decoder.cpp
    src/core/cpu_buffer.cpp
)

target_link_libraries(RTSPCore PRIVATE
    ${GST_LIBRARIES}
    yaml-cpp
    CUDA::cudart
)

target_include_directories(RTSPCore PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GST_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Python Bindings Module
# Note: Source is now in src/bindings/
pybind11_add_module(_core src/bindings/bindings.cpp)
target_link_libraries(_core PRIVATE RTSPCore)

# Output directory strategy
if(SKBUILD)
    # When building via pip/scikit-build, install to the package directory
    # Note: Package directory will likely change to 'src/rtspmodule' or similar in Phase 3
    install(TARGETS _core LIBRARY DESTINATION rtspmodule) 
else()
    # Local development: output to lib/ as before
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/rtspmodule
    )
endif()

# Optional DeepStream SDK support
set(DEEPSTREAM_PATHS
    "/opt/nvidia/deepstream/deepstream"
    "/opt/nvidia/deepstream/deepstream-7.0"
    "/opt/nvidia/deepstream/deepstream-7.1"
    "/opt/nvidia/deepstream/deepstream-8.0"
)

foreach(DS_PATH ${DEEPSTREAM_PATHS})
    if(EXISTS "${DS_PATH}/lib/libnvbufsurface.so")
        message(STATUS "DeepStream SDK detected at ${DS_PATH} - enabling NVMM support")
        target_compile_definitions(RTSPCore PRIVATE HAVE_DEEPSTREAM)
        target_include_directories(RTSPCore PRIVATE ${DS_PATH}/sources/includes)
        find_library(NVBUFSURFACE nvbufsurface PATHS ${DS_PATH}/lib NO_DEFAULT_PATH)
        if(NVBUFSURFACE)
            target_link_libraries(RTSPCore PRIVATE ${NVBUFSURFACE})
            message(STATUS "Linked nvbufsurface: ${NVBUFSURFACE}")
        endif()
        break()
    endif()
endforeach()


