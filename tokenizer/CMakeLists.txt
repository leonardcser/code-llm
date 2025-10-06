cmake_minimum_required(VERSION 3.20)
project(llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Disable C++ extensions")

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto -fomit-frame-pointer -DNDEBUG -Wall -Wextra")

# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

enable_testing()

# Include FetchContent module
include(FetchContent)

# yaml-cpp for parsing yaml files (only needed for C++ executables)
if(NOT BUILD_PYTHON_BINDINGS)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    FetchContent_Declare(
      yaml-cpp
      GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
      GIT_TAG 0.8.0
    )
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests")
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools")
    set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "Build yaml-cpp as static library")
    FetchContent_MakeAvailable(yaml-cpp)
endif()

# cereal (v1.3.2) for binary serialization
FetchContent_Declare(
  cereal
  GIT_REPOSITORY https://github.com/USCiLab/cereal.git
  GIT_TAG v1.3.2
)
set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "Skip cereal performance tests")
set(BUILD_TESTS OFF CACHE BOOL "Skip cereal tests")
set(BUILD_SANDBOX OFF CACHE BOOL "Skip cereal sandbox")
FetchContent_MakeAvailable(cereal)

# RE/flex for fast regex (v6.0.0)
FetchContent_Declare(
    reflex
    GIT_REPOSITORY https://github.com/Genivia/RE-flex.git
    GIT_TAG v6.0.0
)
FetchContent_MakeAvailable(reflex)

# pybind11 for Python bindings (v3.0.1)
if(BUILD_PYTHON_BINDINGS)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Abseil (20250814.1) for data structures
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Propagate C++ standard to absl")
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "Disable absl install")
set(BUILD_TESTING OFF CACHE BOOL "Disable absl tests")
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20250814.1
)
FetchContent_MakeAvailable(absl)

# Define entrypoints (executables)
set(ENTRYPOINTS tokenize visualize decode tokenize-cli)

# Find all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Separate entrypoint files from library sources
set(LIB_SOURCES "")
foreach(SOURCE ${SOURCES})
    get_filename_component(NAME ${SOURCE} NAME_WE)
    if(NOT ${NAME} IN_LIST ENTRYPOINTS AND NOT ${SOURCE} MATCHES ".*_test\\.cpp$" AND NOT ${NAME} STREQUAL "python_bindings")
        list(APPEND LIB_SOURCES ${SOURCE})
    endif()
endforeach()

# Create library from common sources
if(LIB_SOURCES)
    add_library(llm_lib ${LIB_SOURCES})
    target_link_libraries(llm_lib PUBLIC absl::flat_hash_map cereal::cereal ReflexLibStatic)
    target_compile_options(llm_lib PRIVATE -Wno-deprecated-declarations)

    # Disable LTO for library when building Python bindings due to GCC thread-local storage issues
    if(BUILD_PYTHON_BINDINGS)
        target_compile_options(llm_lib PRIVATE -fno-lto)
    endif()
endif()

# Create executables for each entrypoint
if(NOT BUILD_PYTHON_BINDINGS)
    foreach(ENTRYPOINT ${ENTRYPOINTS})
        add_executable(${ENTRYPOINT} src/${ENTRYPOINT}.cpp)
        if(LIB_SOURCES)
            target_link_libraries(${ENTRYPOINT} PRIVATE llm_lib)
        endif()
        target_link_libraries(${ENTRYPOINT} PRIVATE ReflexLibStatic yaml-cpp)
    endforeach()
endif()

# Test executables
if(NOT BUILD_PYTHON_BINDINGS)
    file(GLOB_RECURSE TEST_SOURCES "src/*_test.cpp")
    set(TEST_EXECUTABLES "")
    foreach(TEST_SOURCE ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        list(APPEND TEST_EXECUTABLES ${TEST_NAME})
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        if(LIB_SOURCES)
            target_link_libraries(${TEST_NAME} PRIVATE llm_lib)
        endif()
        target_link_libraries(${TEST_NAME} PRIVATE)
        target_compile_options(${TEST_NAME} PRIVATE -DDEBUG_TESTS -Wno-unused-variable)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()

    # Custom test target
    add_custom_target(test_all
        COMMAND ${CMAKE_CTEST_COMMAND}
        DEPENDS ${TEST_EXECUTABLES}
    )
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(tokenizer_cpp src/python_bindings.cpp)
    if(LIB_SOURCES)
        target_link_libraries(tokenizer_cpp PRIVATE llm_lib)
    endif()
    # Link against static reflex library to avoid runtime dependency issues
    target_link_libraries(tokenizer_cpp PRIVATE ReflexLibStatic)

    # Disable LTO for Python bindings to avoid GCC thread-local storage linker issues
    target_compile_options(tokenizer_cpp PRIVATE -fno-lto)
    target_link_options(tokenizer_cpp PRIVATE -fno-lto)

    # Install the extension module to the package directory
    install(TARGETS tokenizer_cpp LIBRARY DESTINATION tokenizer)
endif()
