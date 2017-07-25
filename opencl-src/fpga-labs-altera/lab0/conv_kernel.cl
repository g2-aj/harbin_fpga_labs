#define IMAGE_HEIGHT   (28)
#define IMAGE_WIDTH    (28)
#define FILTER_SIZE    (3)

__kernel  __attribute__ ((reqd_work_group_size(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
void __conv_2d(
    __global float * restrict in,               // W*H input images
    __global float * restrict filt,             // K*K filter kernel
    __global float * restrict out,              // W*H output images
    const float pBias)                // constant offset/bias
{
    // get image pixel
    int i = get_local_id(0);
    int j = get_local_id(1);

    float sum = 0;
        
    // loop over rows - "pragma unroll 1" directive to not unroll either loop
    #pragma unroll 1
    for (int r = 0; r < FILTER_SIZE; r++) 
    {
        #pragma unroll 1
        for(int c = 0; c < FILTER_SIZE; c++)
        {
            sum += filt[r * FILTER_SIZE + c]*in[(j + r) * IMAGE_WIDTH + i + c];
        }
    }
    out[j * IMAGE_WIDTH + i] = sum + pBias;
}
