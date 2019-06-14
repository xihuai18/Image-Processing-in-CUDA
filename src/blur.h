#ifndef BLUR_H
#define BLUR_H


__global__ void kernel_blur(int S,
                            int img_height, 
                            int img_width, 
                            int *res_img,
                            const int *__restrict__ src_img);

int* imgBlur(int *src_img, int img_height, int img_width, int S = 2);

void calculateGaussKernel(int S);

#endif // BLUR_H