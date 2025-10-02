cmake_minimum_required(VERSION 3.20)
project(llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mcpu=native -funroll-loops -flto -fomit-frame-pointer -DNDEBUG -Wall -Wextra")

enable_testing()

# Include FetchContent module
include(FetchContent)

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
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.1
)
FetchContent_MakeAvailable(pybind11)

find_package(absl CONFIG REQUIRED)

# Define entrypoints (executables)
set(ENTRYPOINTS tokenize visualize decode)

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
endif()

# Create executables for each entrypoint
foreach(ENTRYPOINT ${ENTRYPOINTS})
    add_executable(${ENTRYPOINT} src/${ENTRYPOINT}.cpp)
    if(LIB_SOURCES)
        target_link_libraries(${ENTRYPOINT} PRIVATE llm_lib)
    endif()
    target_link_libraries(${ENTRYPOINT} PRIVATE ReflexLibStatic)
endforeach()

# Test executables
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

# Python bindings
pybind11_add_module(tokenizer_cpp src/python_bindings.cpp)
if(LIB_SOURCES)
    target_link_libraries(tokenizer_cpp PRIVATE llm_lib)
endif()
# Link against static reflex library to avoid runtime dependency issues
target_link_libraries(tokenizer_cpp PRIVATE ReflexLibStatic)

# Install the extension module to the package directory
install(TARGETS tokenizer_cpp LIBRARY DESTINATION tokenizer)
