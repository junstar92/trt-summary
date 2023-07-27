cmake_minimum_required(VERSION 3.18)
project(trtsummary
    LANGUAGES CXX CUDA
)
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# Set Build Type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Find TensorRT Libraries (from `LD_LIBRARY_PATH` environment variable)
set(trt_libraries nvinfer nvonnxparser)
foreach(lib ${trt_libraries})
    find_library(TRT_${lib}_LIBRARY
        NAMES ${lib}
        HINTS ENV LD_LIBRARY_PATH
        REQUIRED
        NO_DEFAULT_PATH
    )
    message(STATUS "Found ${lib} : ${TRT_${lib}_LIBRARY}")
endforeach()

# Find TensorRT Include Directory Path
get_filename_component(TRT_LIBRARY_DIR ${TRT_nvinfer_LIBRARY} DIRECTORY)
find_path(TRT_INCLUDE_DIR
    NAME
        NvInfer.h
    PATHS
        "${TRT_LIBRARY_DIR}/.."
    PATH_SUFFIXES include
    REQUIRED
    NO_DEFAULT_PATH
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/engine_inspector.cpp
    src/onnx_inspector.cpp
    src/jsoncpp.cpp
)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        src
        ${TRT_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${TRT_nvinfer_LIBRARY}
        ${TRT_nvonnxparser_LIBRARY}
        ${CUDA_RUNTIME_LIBRARY}
)
set_target_properties(trtsummary
    PROPERTIES
        CXX_STANDARD 14
)