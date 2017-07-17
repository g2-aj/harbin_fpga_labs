#define IMAGE_HEIGHT   (28)
#define IMAGE_WIDTH    (28)
#define FILTER_SIZE    (3)

/*__kernel  __attribute__ ((reqd_work_group_size(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
void __conv_2d(
    __global float *in,               // W*H input images
    __global float *filt,             // K*K filter kernel
    __global float *out,              // W*H output images
    const float pBias)                // constant offset/bias
{
    // get image resolution
    int i = get_local_id(0);
    int j = get_local_id(1);

    float sum = 0;
        
    // loop over rows
    for (int r = 0; r < FILTER_SIZE; r++) 
    {
       // __attribute__((opencl_unroll_hint))
        for(int c = 0; c < FILTER_SIZE; c++)
        {
            sum += filt[r * FILTER_SIZE + c]*in[(j + r) * IMAGE_WIDTH + i + c];
        }
    }
    out[j * IMAGE_WIDTH + i] = sum + pBias;
}*/

__kernel  __attribute__ ((reqd_work_group_size(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
void conv_2d(
    __global float * restrict in,               // W*H input images
    __constant float * restrict filt,           // K*K filter kernel
    __global float *restrict out,              // W*H output images
    const float pBias)                // constant offset/bias
{
    __local float local_image[IMAGE_WIDTH * IMAGE_HEIGHT];
    __local float local_filt[FILTER_SIZE * FILTER_SIZE];

    int x = get_local_id(0);
    int y = get_local_id(1);
    if(x < FILTER_SIZE*FILTER_SIZE) {
        local_filt[x] = filt[x];
    }
    local_image[y * IMAGE_WIDTH + x] = in[y * IMAGE_WIDTH + x];

    // wait for all work items to copy their share as each work item
    // requires 3x3 neighbor instead of single pixel
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
        
    // loop over rows
    int i = get_local_id(0);
    int j = get_local_id(1);
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

