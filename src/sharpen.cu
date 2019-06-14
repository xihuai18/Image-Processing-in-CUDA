#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "sharpen.h"
#include "common.h"

// the kernel of sobel using sharpening img
int sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
int sobel_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
__constant__ int sobel_kernel_x[9];
__constant__ int sobel_kernel_y[9];

// 使用sobel算子的图像锐化处理，每个线程负责一个像素点
__global__ void kernel_sharpen(int img_height, int img_width, int *res_img,
                               const int *__restrict__ src_img) {
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;

  int i = thread_id / img_width, j = thread_id % img_width;
  if (i >= 1 && i < img_height - 1 && j >= 1 && j < img_width - 1) {
    int kernel_index = 0;
    int sum_x[3] = {0}, sum_y[3] = {0};
    for(int row = i - 1; row <= i + 1;++row) {
      for(int col = j - 1; col <= j + 1; ++col) {
        int src_img_value = src_img[row * img_width + col];
        int sobel_kernel_x_value = sobel_kernel_x[kernel_index];
        int sobel_kernel_y_value = sobel_kernel_y[kernel_index];

        for (int k = 2; k >= 0; --k) {
          sum_x[k] += sobel_kernel_x_value * (src_img_value & 255);
          sum_y[k] += sobel_kernel_y_value * (src_img_value & 255);
          src_img_value >>= 8;
        }
      }
      ++kernel_index;
    }

    int pixel_value = src_img[thread_id];

    int rgb[3] = {0};
    for (int i = 2; i >= 0; --i) {
      rgb[i] = int(sqrt((float)((sum_x[i] << 1) + (sum_y[i] << 1)))) / 8;
      rgb[i] += (pixel_value & 255);
      rgb[i] = rgb[i] < 0 ? 0 : (rgb[i] > 255 ? 255 : rgb[i]);
      pixel_value >>= 8;
    }

    res_img[thread_id] = (rgb[0] << 16) + (rgb[1] << 8) + rgb[2];
  }
}

// 使用sobel算子进行图像的锐化
int* imgSharpen(int *src_img, int img_height, int img_width) {
  // copy to constant memory of device
  cudaMemcpyToSymbol(sobel_kernel_x, sobel_x, sizeof(int) * 9);
  cudaMemcpyToSymbol(sobel_kernel_y, sobel_y, sizeof(int) * 9);

  int img_size = img_height * img_width;
  int img_size_bytes = img_size * sizeof(int);

  int *h_res_img = (int*)malloc(img_size_bytes);

  int *d_src_img = NULL, *d_res_img = NULL;
  cudaMalloc((void**)&d_src_img, img_size_bytes);
  cudaMalloc((void**)&d_res_img, img_size_bytes);

  dim3 block(1024, 1, 1), grid(1, 1, 1);
  if (img_size < 1024) {
    block.x = img_size;
  } else {
    grid.x = updiv(img_size, 1024);
  }

  cudaMemcpy(d_src_img, src_img, img_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res_img, src_img, img_size_bytes, cudaMemcpyHostToDevice);
  kernel_sharpen<<<grid, block>>>(img_height, img_width, d_res_img, d_src_img);
  cudaMemcpy(h_res_img, d_res_img, img_size_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_res_img);
  cudaFree(d_src_img);

  return h_res_img;
}