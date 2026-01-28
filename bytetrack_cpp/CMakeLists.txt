cmake_minimum_required(VERSION 3.18)
project(bytetrack_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_SHARED_LIB "Build shared library" ON)

# Find dependencies
find_package(Eigen3 REQUIRED)

# Header files
set(HEADERS
    include/bytetrack/bytetrack.hpp
    include/bytetrack/data_types.hpp
    include/bytetrack/kalman_filter.hpp
    include/bytetrack/strack.hpp
    include/bytetrack/lapjv.hpp
    include/bytetrack/byte_tracker.hpp
)

# Source files
set(SOURCES
    src/kalman_filter.cpp
    src/strack.cpp
    src/lapjv.cpp
    src/byte_tracker.cpp
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Shared library
if(BUILD_SHARED_LIB)
    add_library(bytetrack SHARED ${SOURCES} ${HEADERS})
    target_include_directories(bytetrack PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(bytetrack PUBLIC Eigen3::Eigen)
    
    # Install library
    install(TARGETS bytetrack
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(DIRECTORY include/bytetrack DESTINATION include)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    
    if(NOT pybind11_FOUND)
        # Fetch pybind11 if not found
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    pybind11_add_module(bytetrack_cpp src/bindings.cpp ${SOURCES})
    target_include_directories(bytetrack_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(bytetrack_cpp PRIVATE Eigen3::Eigen)
    
    # Copy to python package location
    set(PYTHON_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
    add_custom_command(TARGET bytetrack_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:bytetrack_cpp>
            ${PYTHON_OUTPUT_DIR}
        COMMENT "Copying Python module to ${PYTHON_OUTPUT_DIR}"
    )
endif()

# Compiler optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(bytetrack PRIVATE -O3 -march=native -ffast-math)
    if(BUILD_PYTHON_BINDINGS)
        target_compile_options(bytetrack_cpp PRIVATE -O3 -march=native -ffast-math)
    endif()
endif()

