cmake_minimum_required(VERSION 4.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 75 80)

project(miniTensor
        VERSION 0.1
        DESCRIPTION "A simple tensor library"
        LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to `Debug` as none was specified.")
        set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define options for the user, defaulting to OFF
option(USE_CUDA "Enable NVIDIA CUDA backend" OFF)
option(USE_ROCM "Enable AMD ROCm/HIP backend" OFF)

# Attempt auto-detection if the user didn't specify
if(NOT USE_CUDA AND NOT USE_ROCM)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "Auto-detected CUDA Toolkit.")
        set(USE_CUDA ON)
    else()
        # ROCm/HIP detection
        find_package(HIP QUIET)
        if(HIP_FOUND)
            message(STATUS "Auto-detected ROCm/HIP.")
            set(USE_ROCM ON)
        endif()
    endif()
endif()

# Prevent user from enabling both simultaneously (if your architecture requires it)
if(USE_CUDA AND USE_ROCM)
    message(FATAL_ERROR "Cannot enable both CUDA and ROCm simultaneously.")
endif()

add_library(tensor "src/gpu_ops.cpp")
target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(USE_CUDA)
    target_include_directories(tensor PUBLIC "/opt/cuda/include")
elseif(USE_ROCM)
    target_include_directories(tensor PUBLIC "/opt/rocm/include")
endif()

find_package(GTest REQUIRED)

add_executable(tensor_tests test/tensor_test.cpp)
target_include_directories(tensor_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Configuration based on the selected backend
if(USE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(tensor PUBLIC -DUSE_CUDA)
    set_source_files_properties(
        src/gpu_ops.cpp 
        test/tensor_test.cpp 
        PROPERTIES LANGUAGE CUDA
    )
elseif(USE_ROCM)
    enable_language(HIP)
    target_compile_definitions(tensor PUBLIC -DUSE_ROCM)
    set_source_files_properties(
        src/gpu_ops.cpp 
        test/tensor_test.cpp 
        PROPERTIES LANGUAGE HIP
    )
else()
    message(STATUS "No GPU backend selected. Falling back to CPU only.")
endif()

target_link_libraries(tensor_tests
    PRIVATE
    tensor
    GTest::gtest
    GTest::gtest_main
)
