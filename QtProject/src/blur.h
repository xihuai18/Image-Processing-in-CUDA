#ifndef BLUR_H
#define BLUR_H

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_blur(int S,
                            int img_height, 
                            int img_width, 
                            int *res_img,
                            const int *__restrict__ src_img);
extern "C"
int* imgBlur(int *src_img, int img_height, int img_width);

void calculateGaussKernel(int S);

#endif // BLUR_H
