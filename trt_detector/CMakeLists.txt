cmake_minimum_required(VERSION 3.18)
project(trt_detector LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

# TensorRT
if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include
    PATHS /usr/local/tensorrt /usr/include/x86_64-linux-gnu
)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib lib64
    PATHS /usr/local/tensorrt /usr/lib/x86_64-linux-gnu
)

message(STATUS "TensorRT: ${TENSORRT_LIBRARY}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")

set(SOURCES
    src/preprocessor.cpp
    src/postprocessor.cpp
    src/trt_engine.cpp
    src/detector_service.cpp
    src/async_pipeline.cpp
    src/bindings.cpp
    src/cuda_preprocess.cu
)

pybind11_add_module(trt_detector ${SOURCES})

set_target_properties(trt_detector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(trt_detector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(trt_detector PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
)

install(TARGETS trt_detector LIBRARY DESTINATION .)

