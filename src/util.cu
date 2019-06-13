#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "build_graph.h"
#define PI 3.141593

// the kernel of sobel using sharpening img
__constant__ int sobel_kernel_x[9];
__constant__ int sobel_kernel_y[9];

// the kernel of gauss using bluring img
__constant__ float gauss_kernel[1000];

__global__ void kernel_sharpen(int img_height, int img_width, int *res_img,
                               const int *__restrict__ src_img) {
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;

  if (thread_id < img_width * img_height) {
    int i = thread_id / img_width, j = thread_id % img_width;
    int kernel_index = 0;
    int sum_x[3] = {0}, sum_y[3] = {0};
    for(int row = i - 1; row <= i + 1;++row) {
        int data_index = row * img_width + (j - 1);
        for(int col = j - 1; col <= j + 1; ++col) {
            // 判断是否越界，把越界的像素点值当成0处理
            if (row >= 0 && row < img_height && col >= 0 && col < img_width) {
              int src_img_value = src_img[data_index];
              int sobel_kernel_x_value = sobel_kernel_x[kernel_index];
              int sobel_kernel_y_value = sobel_kernel_y[kernel_index];

              for (int i = 0; i < 3; ++i) {
                sum_x[i] += sobel_kernel_x_value * (src_img_value & 255);
                sum_y[i] += sobel_kernel_y_value * (src_img_value & 255);
                src_img_value >>= 8;
              }
            }
            ++kernel_index;
            ++data_index;
        }
    }
    
    int rgb[3] = {0};
    for (int i = 0; i < 3; ++i) {
      rgb[i] = sqrt((float)((sum_x[i] << 2) + (sum_y[i] << 2)));
    }

    res_img[thread_id] = (rgb[2] << 16) + (rgb[1] << 8) + rgb[0];
  }
}

// 使用sobel算子进行图像的锐化
int* imgSharpen(int *src_img, int img_height, int img_width) {
  int sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  int sobel_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
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
  kernel_sharpen<<<grid, block>>>(img_height, img_width, d_res_img, d_src_img);
  cudaMemcpy(h_res_img, d_res_img, img_size_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_res_img);
  cudaFree(d_src_img);

  return h_res_img;
}

__global__ void kernel_blur(int img_height, int img_width, int *res_img,
                            const int *__restrict__ src_img) {
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;

  int S = gauss_kernel_s;

  if (thread_id < img_width * img_height) {
    int i = thread_id / img_width, j = thread_id % img_width;
    int gauss_index = 0;
    float sum[3] = {0};
    for(int row= i - 3 * S; row <= i + 3 * S; ++row) {
      int data_index = row * img_width + (j - 3 * S);
      for(int col = j - 3 * S; col <= j + 3 * S; ++col) {
        // 判断是否越界，把越界的像素点值当成0处理
        if (row >= 0 && row < img_height && col >= 0 && col < img_width) {
          int src_img_value = src_img[data_index];
          float gauss_kernel_value = gauss_kernel[gauss_index];

          for (int i = 0; i < 3; ++i) {
            sum[i] += gauss_kernel_value * (src_img_value & 255);
            src_img_value >>= 8;
          }
        }
        ++gauss_index;
        ++data_index;
      }
    }
    
    res_img[thread_id] = (int(sum[2]) << 16) + (int(sum[1]) << 8) + int(sum[0]);
  }
}

// 使用高斯核进行图像模糊处理
int* imgBlur(int *src_img, int img_height, int img_width) {
  calculateGaussKernel();

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
  kernel_blur<<<grid, block>>>(img_height, img_width, d_res_img, d_src_img);
  cudaMemcpy(h_res_img, d_res_img, img_size_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_res_img);
  cudaFree(d_src_img);

  return h_res_img;
}

void calculateGaussKernel() {
  int S = gauss_kernel_s;
  int n = 6 * S + 1;
  int size = sizeof(float) * n * n;
  float *h_gauss_kernel = (float*)malloc(size);
  
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int x = i - 3 * S, y = j - 3 * S;
      h_gauss_kernel[i * n + j] = 1 / (S * sqrt(2 * PI)) * 
                                    exp(-1.0 * (x * x + y * y) / ( 2 * S * S));
    }
  }
    
  // 将计算的结果拷贝到cuda constant内存里
  cudaMemcpyToSymbol(gauss_kernel, h_gauss_kernel, size);
}