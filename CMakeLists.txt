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
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra")

enable_testing()

# Define entrypoints (executables)
set(ENTRYPOINTS main)

# Find all source files
file(GLOB SOURCES "src/*.cpp")

# Separate entrypoint files from library sources
set(LIB_SOURCES "")
foreach(SOURCE ${SOURCES})
    get_filename_component(NAME ${SOURCE} NAME_WE)
    if(NOT ${NAME} IN_LIST ENTRYPOINTS AND NOT ${SOURCE} MATCHES ".*_test\\.cpp$")
        list(APPEND LIB_SOURCES ${SOURCE})
    endif()
endforeach()

# Create library from common sources
if(LIB_SOURCES)
    add_library(llm_lib ${LIB_SOURCES})
endif()

# Create executables for each entrypoint
foreach(ENTRYPOINT ${ENTRYPOINTS})
    add_executable(${ENTRYPOINT} src/${ENTRYPOINT}.cpp)
    if(LIB_SOURCES)
        target_link_libraries(${ENTRYPOINT} llm_lib)
    endif()
endforeach()

# Test executables
file(GLOB TEST_SOURCES "src/*_test.cpp")
set(TEST_EXECUTABLES "")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    list(APPEND TEST_EXECUTABLES ${TEST_NAME})
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    if(LIB_SOURCES)
        target_link_libraries(${TEST_NAME} llm_lib)
    endif()
    target_compile_options(${TEST_NAME} PRIVATE -DDEBUG_TESTS -Wno-unused-variable)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# Custom test target
add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS ${TEST_EXECUTABLES}
)
