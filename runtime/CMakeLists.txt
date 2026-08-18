# NpuEmbeddings -- M7 C++ runtime.
# SPDX-License-Identifier: Apache-2.0
#
# TRAP, from CLAUDE.md and mlir-aie #3048:
#   project() MUST come before find_package(XRT), or linking silently
#   downgrades to static. Silently -- the build succeeds and the binary is
#   wrong. Do not reorder these two lines.
cmake_minimum_required(VERSION 3.20)
project(npuembed_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# XRT lives at C:\Xilinx\XRT on this machine, NOT under Program Files, and
# XILINX_XRT must stay unset because it poisons Windows builds (iron_setup.py).
# XRT_ROOT is the variable to use.
if(NOT DEFINED XRT_ROOT)
  if(DEFINED ENV{XRT_ROOT})
    set(XRT_ROOT $ENV{XRT_ROOT})
  else()
    set(XRT_ROOT "C:/Xilinx/XRT")
  endif()
endif()
message(STATUS "XRT_ROOT: ${XRT_ROOT}")

find_path(XRT_INCLUDE_DIR xrt/xrt_device.h
          HINTS "${XRT_ROOT}/include" REQUIRED)
find_library(XRT_COREUTIL_LIB NAMES xrt_coreutil
             HINTS "${XRT_ROOT}/lib" "${XRT_ROOT}/lib/Release" REQUIRED)
message(STATUS "XRT headers: ${XRT_INCLUDE_DIR}")
message(STATUS "XRT lib:     ${XRT_COREUTIL_LIB}")

# Boost is only needed if the compiler reports pre-C++17, which on MSVC means
# /Zc:__cplusplus was not passed (see below). With that flag XRT uses std::any
# and this is unused -- kept as a fallback so the build works either way.
if(NOT DEFINED BOOST_ROOT AND EXISTS "C:/dev/boost_1_88_0")
  set(BOOST_ROOT "C:/dev/boost_1_88_0")
endif()
if(DEFINED BOOST_ROOT AND EXISTS "${BOOST_ROOT}")
  message(STATUS "Boost (fallback only): ${BOOST_ROOT}")
endif()

add_executable(npuembed
  src/main.cpp
  src/npue.cpp
    src/tokenizer.cpp
  src/npue_pack.cpp
  src/npu_device.cpp
)
target_include_directories(npuembed PRIVATE include "${XRT_INCLUDE_DIR}")
if(DEFINED BOOST_ROOT AND EXISTS "${BOOST_ROOT}")
  target_include_directories(npuembed PRIVATE "${BOOST_ROOT}")
endif()
# ws2_32: the /v1/embeddings endpoint (include/http.hpp) talks Winsock
target_link_libraries(npuembed PRIVATE "${XRT_COREUTIL_LIB}" ws2_32)

if(MSVC)
  # /Zc:__cplusplus is REQUIRED, not cosmetic.
  #
  # xrt/detail/any.h picks std::any when __cplusplus >= 201703L and falls back
  # to boost/any.hpp otherwise. MSVC reports __cplusplus as 199711L unless this
  # flag is given -- even under -std:c++17 -- so without it the build demands
  # Boost and fails with:
  #     fatal error C1083: Cannot open include file: 'boost/any.hpp'
  # which reads as a missing dependency and is really a missing flag.
  # /O2 explicitly: a Release-less single-config generator would otherwise build
  # this unoptimised, and the host-side bf16 conversion is 13.8 M elements per
  # encode. /arch:AVX2 enables the vectorised conversion in main.cpp -- Strix
  # Point is Zen 5, so AVX2 is a floor, not an assumption.
  target_compile_options(npuembed PRIVATE /W3 /EHsc /Zc:__cplusplus /O2 /arch:AVX2)
  # XRT headers use std::min/max; the Windows macros break them. XRT's own
  # headers also define NOMINMAX, so let theirs win rather than warning.
  target_compile_definitions(npuembed PRIVATE WIN32_LEAN_AND_MEAN)
endif()
