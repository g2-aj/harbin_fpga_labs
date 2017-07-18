/*******************************************************************************
Vendor: Xilinx
Associated Filename: main.c
#Purpose: An example showing new kernels can be downloaded to FPGA while keeping
#         the data in device memory intact
#*******************************************************************************
Copyright (c) 2016, Xilinx, Inc.^M
All rights reserved.^M
^M
Redistribution and use in source and binary forms, with or without modification, ^M
are permitted provided that the following conditions are met:^M
^M
1. Redistributions of source code must retain the above copyright notice, ^M
this list of conditions and the following disclaimer.^M
^M
2. Redistributions in binary form must reproduce the above copyright notice, ^M
this list of conditions and the following disclaimer in the documentation ^M
and/or other materials provided with the distribution.^M
^M
3. Neither the name of the copyright holder nor the names of its contributors ^M
may be used to endorse or promote products derived from this software ^M
without specific prior written permission.^M
^M
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ^M
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, ^M
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. ^M
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, ^M
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, ^M
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) ^M
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, ^M
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, ^M
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "pgm.h"

using namespace aocl_utils;
using namespace std;
////////////////////////////////////////////////////////////////////////////////

#define FILTER_SIZE    (3)
typedef float IMG_DTYPE;
////////////////////////////////////////////////////////////////////////////////

void print_matrix(int *mat, int n_rows, int n_cols) {
    for(int r = 0; r < n_rows; r++) {
        for(int c = 0; c < n_cols; c++) {
            printf("%d, ", mat[r*n_cols+c]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
    int test_fail = 0;
    int row, col, pix;
    pgm_t input_img, output_img;

    IMG_DTYPE filter[FILTER_SIZE*FILTER_SIZE] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    IMG_DTYPE *h_input;      // input image buffer
    IMG_DTYPE *hw_output;    // host buffer for device output
    IMG_DTYPE *sw_output;    // host buffer for reference output

    size_t global[2];                   // global domain size for our calculation
    size_t local[2];                    // local domain size for our calculation

    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    cl_mem d_in_image;                  // device buffer for input image
    cl_mem d_in_filter;                 // device buffer for filter kernel
    cl_mem d_out_image;                 // device buffer for filtered image

    printf("Application start\n");
    const double start_time = getCurrentTimestamp();
    if (argc != 3) {
        printf("Usage: %s conv_kernel image_path/image_name.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *img_name = argv[2];
    // read the image and initialize the host buffer with that
    err = readPGM(&input_img, img_name);
    if(err < 0) {
        printf("Cound not read the image\n");
        return EXIT_FAILURE;
    }

    printf("Input image resolution = %dx%d\n", input_img.width, input_img.height);
    h_input = (IMG_DTYPE*)malloc(sizeof(IMG_DTYPE)*input_img.height*input_img.width); 
    hw_output = (IMG_DTYPE*)malloc(sizeof(IMG_DTYPE)*input_img.height*input_img.width); 
    sw_output = (IMG_DTYPE*)malloc(sizeof(IMG_DTYPE)*input_img.height*input_img.width); 
    for(pix = 0; pix < input_img.height*input_img.width; pix++) {
        h_input[pix] = input_img.buf[pix];
    }

    // Connect to first platform
    err = clGetPlatformIDs(1,&platform_id,NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to find an OpenCL platform!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS) {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("INFO: CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
    if (err != CL_SUCCESS) {
        printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("INFO: CL_PLATFORM_NAME %s\n",cl_platform_name);

    // Connect to a compute device
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
                         1, &device_id, NULL);
    if (err != CL_SUCCESS) {
            printf("Error: Failed to create a device group!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
        printf("Error: code %i\n",err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    int status;

    // Create Program Objects

    // Load binary from disk
    std::string binary_file = getBoardBinaryFile(argv[1], device_id);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device_id, 1);

    if ((!program) || (err!=CL_SUCCESS)) {
        printf("Error: Failed to create compute program from binary %d!\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "__conv_2d", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create the input and output arrays in device memory for our calculation
    d_in_image = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(IMG_DTYPE) * input_img.height*input_img.width, NULL, NULL);
    d_in_filter = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(IMG_DTYPE) * FILTER_SIZE * FILTER_SIZE, NULL, NULL);
    d_out_image = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  sizeof(IMG_DTYPE) * input_img.height*input_img.width, NULL, NULL);
    if (!d_in_image || !d_in_filter || !d_out_image) {
        printf("Error: Failed to allocate device memory!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Write the image from host buffer to device memory
    err = clEnqueueWriteBuffer(commands, d_in_image, CL_TRUE, 0, sizeof(IMG_DTYPE) * input_img.height*input_img.width, h_input, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to image to device memory!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    // Write filter kernel into device buffer
    //
    err = clEnqueueWriteBuffer(commands, d_in_filter, CL_TRUE, 0, sizeof(IMG_DTYPE) * FILTER_SIZE * FILTER_SIZE, filter, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to filter coeff into device memory!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Set the arguments to our compute kernel
    int filter_size = FILTER_SIZE;
    IMG_DTYPE bias = 1;
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in_image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_in_filter);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_out_image);
    err |= clSetKernelArg(kernel, 3, sizeof(IMG_DTYPE),    &bias);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    
    // get the work group info
    size_t wg_info[3];
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(wg_info), wg_info, NULL);
    printf("WG Info : %lu, %lu, %lu\n", wg_info[0], wg_info[1], wg_info[2]);
    // Launch computation kernel
    global[0] = wg_info[0];
    global[1] = wg_info[1];
    local[0] = wg_info[0];
    local[1] = wg_info[1];
    
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, (size_t*)&global, (size_t*)&local, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Read back the results from the device to verify the output
    cl_event readevent;
    err = clEnqueueReadBuffer(commands, d_out_image, CL_TRUE, 0,
        sizeof(IMG_DTYPE) * input_img.width*input_img.height, hw_output, 0, NULL, &readevent);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    clWaitForEvents(1, &readevent);
    const double end_time = getCurrentTimestamp();
    const double total_time = end_time - start_time;

    // Generate reference output
    int kr, kc;
    IMG_DTYPE sum = 0;
    for(row = 0; row < input_img.height-FILTER_SIZE+1; row++) {
        for(col = 0; col < input_img.width-FILTER_SIZE+1; col++) {
            sum = 0;
            for(kr = 0; kr < FILTER_SIZE; kr++) {
                for(kc = 0; kc < FILTER_SIZE; kc++ ) {
                    sum += (filter[kr*FILTER_SIZE + kc] * h_input[(row+kr)*input_img.width + col + kc]);
                }
            }
            sw_output[row*input_img.width + col] = sum + bias;
        }
    }
    // Check Results
    for(row = 0; row < input_img.height-FILTER_SIZE+1; row++) {
        for(col = 0; col < input_img.width-FILTER_SIZE+1; col++) {
             if(sw_output[row*input_img.width+col] != hw_output[row*input_img.width+col]){
                 printf("Mismatch at : row = %d, col = %d, expected = %f, got = %f\n",
                     row, col, sw_output[row*input_img.width+col], hw_output[row*input_img.width+col]);
                 test_fail = 1;
             }
        }
    }
    printf("---------Input image-----------\n");
    //print_matrix(h_input, input_img.height, input_img.width);
    printf("---------Reference output------\n");
    //print_matrix(sw_output, input_img.height, input_img.width);
    printf("---------OCL Kernel output-----\n");
    //print_matrix(hw_output, input_img.height, input_img.width);

    // store the output image
    output_img.width = input_img.width;
    output_img.height = input_img.height;
    normalizeF2PGM(&output_img, hw_output);
    writePGM(&output_img, "fpga_output.pgm");
    cout << "Time(ms): " << total_time*1e3 << endl;
    //--------------------------------------------------------------------------
    // Shutdown and cleanup
    //--------------------------------------------------------------------------
    clReleaseMemObject(d_in_image);
    clReleaseMemObject(d_in_filter);
    clReleaseMemObject(d_out_image);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    destroyPGM(&input_img);
    if (test_fail) {
        printf("INFO: TEST FAILED !!!!\n");
        //return EXIT_FAILURE;
    } else {
        printf("INFO: ****TEST PASSED****\n");
    }
}
                                                    
