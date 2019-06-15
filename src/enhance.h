#ifndef ENHANCE_H
#define ENHANCE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define THRESHOLD 64
#define TILESIZE 16

texture<int, 2, cudaReadModeElementType> tex1;
texture<int, 2, cudaReadModeElementType> tex2;

__device__ void RGB2HSI(int *rgb_img, int height, int width);
__device__ void HSI2RGB(int *hsi_img, int height, int width);
__global__ void CLAHE(int *img, int height, int width);
// the 'grid' size is 1/4 of the block size
__global__ void CLAHEPre(int *hsi_img, int *g_frq, int height, int width);
__global__ void CLAHEAft(int *hsi_img, int *g_frq, int height, int width);

int *imgCLAHE(int *src_img, int img_height, int img_width);
int *imgCLAHE_Global(int *src_img, int img_height, int img_width);

#endif  // ENHANCE_H