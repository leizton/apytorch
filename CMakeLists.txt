cmake_minimum_required(VERSION 3.13)
project(pytorch)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)
include_directories(./aten/src)

file(GLOB pytorch_src
        "c10/*/*.h"
        "c10/*/*.cc"
        "c10/*/*/*.h"
        "c10/*/*/*.cc"
        "c10/*/*/*/*.h"
        "c10/*/*/*/*.cc"
        "caffe2/*/*.h"
        "caffe2/*/*.cc"
        "caffe2/*/*/*.h"
        "caffe2/*/*/*.cc"
        "caffe2/*/*/*/*.h"
        "caffe2/*/*/*/*.cc"
        "torch/csrc/*/*.h"
        "torch/csrc/*/*.cc"
        "torch/csrc/*/*/*.h"
        "torch/csrc/*/*/*.cc"
        "torch/csrc/*/*/*/*.h"
        "torch/csrc/*/*/*/*.cc"
        "aten/src/ATen/*.h"
        "aten/src/ATen/*.cc"
        "aten/src/ATen/*/*.h"
        "aten/src/ATen/*/*.cc"
        "aten/src/ATen/*/*/*.h"
        "aten/src/ATen/*/*/*.cc"
        "aten/src/ATen/*/*/*/*.h"
        "aten/src/ATen/*/*/*/*.cc"
)

add_executable(pytorch_bin ${pytorch_src})
