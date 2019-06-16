#ifndef SHARPEN_H
#define SHARPEN_H

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_sharpen(int img_height, int img_width, int *res_img,
                               const int *__restrict__ src_img);
extern "C"
int* imgSharpen(int *src_img, int img_height, int img_width);

#endif // SHARPEN_H
