/*
 * @Author: X Wang, Y xiao, Ch Yang, G Ye
 * @Date: 2019-06-17 01:03:01
 * @Last Modified by: X Wang, Y Xiao, Ch Yang, G Ye
 * @Last Modified time: 2019-06-17 10:51:30
 * @file description:
    blur image using gauss kernel
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "blur.h"
#include "common.h"
#define PI 3.141593

// The kernel of gauss using bluring img;
// And the size of gauss kernel is in [1, 8];
__constant__ float gauss_kernel[2500];

/*
  图像模糊处理，每个线程负责一个像素点，S为高斯核大小
*/
__global__ void kernel_blur(int S, int img_height, int img_width, int *res_img,
                            const int *__restrict__ src_img) {
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;

  int i = thread_id / img_width, j = thread_id % img_width;

  if (thread_id < img_height * img_width) {
    int gauss_index = 0;
    float rgb[3] = {0, 0, 0};
    for (int row = i - 3 * S; row <= i + 3 * S; ++row) {
      for (int col = j - 3 * S; col <= j + 3 * S; ++col) {
        int src_img_value;
        float gauss_kernel_value = gauss_kernel[gauss_index++];

        // 如果该点没越界
        if (row >= 0 && row < img_height && col >= 0 && col < img_width) {
          src_img_value = src_img[row * img_width + col];

          // 如果该点越界，取该点与中心对称的点
        } else {
          int reflect_row = i + (i - row);
          int reflect_col = j + (j - col);
          src_img_value = src_img[reflect_row * img_width + reflect_col];
        }

        for (int k = 2; k >= 0; --k) {
          rgb[k] += gauss_kernel_value * (src_img_value & 255);
          src_img_value >>= 8;
        }
      }
    }

    for (int i = 0; i < 3; ++i) {
      rgb[i] = rgb[i] < 0 ? 0 : (rgb[i] > 255 ? 255 : rgb[i]);
    }

    res_img[thread_id] = (int(rgb[0]) << 16) + (int(rgb[1]) << 8) + rgb[2];
  }
}

/*
  使用高斯核进行图像模糊处理
  Return::
    @Int array: the result image pixel array after blur
*/
int *imgBlur(int *src_img, int img_height, int img_width) {
  int S = 2;
  calculateGaussKernel(S);

  int img_size = img_height * img_width;
  int img_size_bytes = img_size * sizeof(int);

  int *h_res_img = (int *)malloc(img_size_bytes);

  int *d_src_img = NULL, *d_res_img = NULL;
  cudaMalloc((void **)&d_src_img, img_size_bytes);
  cudaMalloc((void **)&d_res_img, img_size_bytes);

  dim3 block(1024, 1, 1), grid(1, 1, 1);
  if (img_size < 1024) {
    block.x = img_size;
  } else {
    grid.x = updiv(img_size, 1024);
  }

  cudaMemcpy(d_src_img, src_img, img_size_bytes, cudaMemcpyHostToDevice);
  kernel_blur<<<grid, block>>>(S, img_height, img_width, d_res_img, d_src_img);
  cudaMemcpy(h_res_img, d_res_img, img_size_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_res_img);
  cudaFree(d_src_img);

  return h_res_img;
}

/*
  计算高斯核，S为高斯核大小
*/
void calculateGaussKernel(int S) {
  int n = 6 * S + 1;
  int size = sizeof(float) * n * n;
  float *h_gauss_kernel = (float *)malloc(size);

  float sum = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int x = i - 3 * S, y = j - 3 * S;
      h_gauss_kernel[i * n + j] =
          1 / (S * sqrt(2 * PI)) * exp(-1.0 * (x * x + y * y) / (2 * S * S));
      sum += h_gauss_kernel[i * n + j];
    }
  }

  // 对高斯核进行归一化
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      h_gauss_kernel[i * n + j] /= sum;
    }
  }

  // 将计算的结果拷贝到cuda constant内存里
  cudaMemcpyToSymbol(gauss_kernel, h_gauss_kernel, size);
}
