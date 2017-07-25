#define IMAGE_HEIGHT   (28)
#define IMAGE_WIDTH    (28)
#define FILTER_SIZE    (3)

__kernel  __attribute__ ((reqd_work_group_size(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
void conv_2d(
    __global float * restrict in,               // W*H input images
    __constant float * restrict filt,           // K*K filter kernel
    __global float *restrict out,              // W*H output images
    const float pBias)                // constant offset/bias
{

    // TODO: Specify an on-chip (local) memory for the image (see handout)
    // TODO: Specify an on-chip (local) memory for the filter (see handout)

    int x = get_local_id(0);
    int y = get_local_id(1);

    // TODO: Uncomment the following block of code which copies one pixel
    // from the global memory into the on-chip memory specified above
    // Note: each work item copies a single pixel into the local memory
    // and waits for all other work items to finish copying their respective
    // pixels
    /*if(x < FILTER_SIZE*FILTER_SIZE) {
        local_filt[x] = filt[x];
    }
    local_image[y * IMAGE_WIDTH + x] = in[y * IMAGE_WIDTH + x];
    */
    
    // wait for all work items to copy their share as each work item
    // requires 3x3 neighbor instead of single pixel
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
        
    // loop over rows
    int i = get_local_id(0);
    int j = get_local_id(1);

    // "#pragma unroll" directive to fully unroll both loops
    #pragma unroll
    for (int r = 0; r < FILTER_SIZE; r++) 
    {
         #pragma unroll
         for(int c = 0; c < FILTER_SIZE; c++)
         {
            sum += local_filt[r * FILTER_SIZE + c]*local_image[(j + r) * IMAGE_WIDTH + i + c];
         }
    }
    out[j * IMAGE_WIDTH + i] = sum + pBias;
}

