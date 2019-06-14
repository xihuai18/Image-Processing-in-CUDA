#ifndef ENHANCE_H
#define ENHANCE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define THRESHOLD 64
#define TILESIZE 16

__device__ float * RGB2HSI(int * rgb_img, int height, int width);
__device__ int * HSI2RGB(int * hsi_img, int height, int width);
__global__ void CLAHE(int * img, int height, int width);
// the 'grid' size is 1/4 of the block size

int* imgCLAHE(int *src_img, int img_height, int img_width);



#endif // ENHANCE_H