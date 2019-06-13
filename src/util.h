#ifndef UTIL_H
#define UTIL_H

const int gauss_kernel_s = 1;

__global__ void kernel_sharpen(int img_height, int img_width, int *res_img,
                               const int *__restrict__ src_img);

__global__ void kernel_blur(int img_height, int img_width, int *res_img,
                            const int *__restrict__ src_img);

int* imgSharpen(int *src_img, int img_height, int img_width);

int* imgBlur(int *src_img, int img_height, int img_width);

void calculateGaussKernel();

#endif