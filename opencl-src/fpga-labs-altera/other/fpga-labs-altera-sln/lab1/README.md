## Description
This lab involves implementation of simple 2D convolution kernel commanly used in Convolutional Neural Networks(CNN) using OpenCL for Altera FPGAs. Starting with the un-optimized implementation from lab0, we will explore the simple **loop unrolling and local memory optimizations**.

- This implementation is fixed for image and filter size, because we will buffer the entire image into on-chip(local)memory and then perform computations. Hence, the max image resolution is limited by the FPGA resources.
- The local work size is equal to gloal work size in all dimensions because we copy the entire image in a single work group.
- Each work item will copy a single pixel from the global memory into the local memory and wait for all other work items to finish copying their respective pixels.
- The read process is pipelined across all work items and thus the AOCL tool can optimize the HW to issue outstanding memory requests.
- The computation involves reading the pixels from 3x3 window and performing dot-product. Even the compute loop is unrolled and pipelined to perform parallel and pipelined computations within the work item. For example, reading pixels from the local BRAM can be overlapped with the MAC operation. Pipelining hint allows the tool to generate such hardware.

## Steps to run
## Kernel emulation
1. Set EMU=1 in ~/.bashrc
2. Source ~/.bashrc
3. ./compile_kernels.sh to compile the kernel for emulation (s5_ref emulation target)
4. make run-emu (compiles the host application code and runs with a test image)A

## Kernel HW
1. Set EMU=0 in ~/.bashrc
2. Source ~/.bashrc
3. ./compile_kernels.sh hw to compile the kernel for de1soc_sharedonly target
4. The --report option dumps an HTML report where you can view system area and latency

