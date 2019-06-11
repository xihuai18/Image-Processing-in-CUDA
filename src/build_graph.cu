// #include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "onecut_kernel.h"

int updiv(int x, int y) { return (x + y - 1) / y; }

__device__ float sigma_square = 0;

__device__ void convertToRGB(int pixel_value, int *r, int *g, int *b) {
  *b = pixel_value & 255;
  pixel_value >>= 8;
  *g = pixel_value & 255;
  pixel_value >>= 8;
  *r = pixel_value & 255;
}

__device__ int Di(int pixel_p, int pixel_q) {
  int p_r, p_g, p_b;
  int q_r, q_g, q_b;
  convertToRGB(pixel_p, &p_r, &p_g, &p_b);
  convertToRGB(pixel_q, &q_r, &q_g, &q_b);
  return (p_r - q_r) * (p_r - q_r) + (p_g - q_g) * (p_g - q_g) +
         (p_b - q_b) * (p_b - q_b);
}

__device__ void warpReduce(volatile int *sigma_sum, int tid, int block_dim_x) {
  sigma_sum[tid] += tid + 32 >= block_dim_x ? 0 : sigma_sum[tid + 32];
  sigma_sum[tid] += tid + 16 >= block_dim_x ? 0 : sigma_sum[tid + 16];
  sigma_sum[tid] += tid + 8 >= block_dim_x ? 0 : sigma_sum[tid + 8];
  sigma_sum[tid] += tid + 4 >= block_dim_x ? 0 : sigma_sum[tid + 4];
  sigma_sum[tid] += tid + 2 >= block_dim_x ? 0 : sigma_sum[tid + 2];
  sigma_sum[tid] += tid + 1 >= block_dim_x ? 0 : sigma_sum[tid + 1];
}

__global__ void computeSigmaSquareSum(int img_width, int img_height,
                                      const int *__restrict__ src_img) {
  extern __shared__ int sigma_sum[];

  int tid = threadIdx.x;
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;

  int img_size = img_width * img_height;
  sigma_sum[tid] = 0;
  if (thread_id * 2 < img_size) {
    int p_idx = thread_id * 2;
    int pixel_p = src_img[p_idx];
    int p_x = p_idx / img_width;

    if (p_x + 1 < img_height) {  // p-down
      sigma_sum[tid] += Di(pixel_p, src_img[p_idx + img_width]);
    }

    if (p_idx + 1 < img_size) {  // q is valid
      int pixel_q = src_img[p_idx + 1];
      int q_x = (p_idx + 1) / img_width, q_y = (p_idx + 1) % img_width;

      if (p_x == q_x) {  // p-right
        sigma_sum[tid] += Di(pixel_p, pixel_q);
      }

      if (q_y + 1 < img_width) {  // q-right
        sigma_sum[tid] += Di(pixel_q, src_img[p_idx + 2]);
      }

      if (q_x + 1 < img_height) {  // q-down
        sigma_sum[tid] += Di(pixel_q, src_img[p_idx + 1 + img_width]);
      }
    }
  }

  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sigma_sum[tid] += sigma_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warpReduce(sigma_sum, tid, blockDim.x);
  }

  if (tid == 0) {
    atomicAdd(&sigma_square, sigma_sum[0]);
  }
}

__host__ void computeSigmaSquare(int img_height, int img_width,
                                 int *d_src_img) {
  int img_size = img_height * img_width;

  // compute the square of sigma
  dim3 grid(1, 1, 1), block(1024, 1, 1);
  if (img_size < 1024 * 2) {
    block.x = updiv(img_size, 2);
  } else {
    grid.x = updiv(img_size, 1024 * 2);
  }

  computeSigmaSquareSum<<<grid, block, block.x * sizeof(int)>>>(
      img_width, img_height, d_src_img);

  int N = (img_height - 1) * img_width + (img_width - 1) * img_height;
  float h_sigma_square;
  cudaMemcpyFromSymbol((void *)&h_sigma_square, sigma_square, sizeof(float));

  h_sigma_square /= N;

  // printf("sigma_square = %lf\n", h_sigma_square);

  cudaMemcpyToSymbol(sigma_square, (void *)&h_sigma_square, sizeof(float));
}

__device__ float gaussian(int di, float lambda, float sigma_square) {
  return lambda * exp(-di / (2 * sigma_square));
}

__device__ int getColorBinIdx(int pixel_value, int color_bin_size) {
  int r, g, b;
  convertToRGB(pixel_value, &r, &g, &b);

  int per_bin_channel = 256 / color_bin_size;
  return (r / color_bin_size) * per_bin_channel * per_bin_channel +
         (g / color_bin_size) * per_bin_channel + (b / color_bin_size);
}

__global__ void computeEdges(float lambda, float beta, float *edges,
                             int img_width, int img_height, int color_bin_size,
                             const int *__restrict__ src_img,
                             const int *__restrict__ mask_img) {
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;

  // int color_bin_size = pow(256/color_bin_size, 3);
  // int color_bin_size = 256/color_bin_size;
  int img_size = img_height * img_width;
  int edges_width = 6 + 2 + 2;

  if (thread_id < img_size) {
    int idx = thread_id * (edges_width);
    for (unsigned int i = 0; i < edges_width; ++i) {
      edges[idx + i] = 0;
    }

    // add s-t-links or t-t-links
    int seed_value = mask_img[thread_id];
    if (seed_value == 255 << 16) {  // s-t-links
      edges[idx] = edges[idx + 8] = 1000;
    } else if (seed_value == 255 << 8) {  // t-t-links
      edges[idx + 1] = 1000;
    }

    // add a-link of color bins
    int color_bin_idx = getColorBinIdx(src_img[thread_id], color_bin_size);
    edges[idx + 5 + 1] = color_bin_idx;
    edges[idx + 5 + 2] = edges[idx + 9] = beta;

    // add n-links
    int pixel_p = src_img[thread_id];
    if (thread_id % img_width + 1 < img_width) {  // right
      edges[idx + 5] = edges[idx + edges_width + 4] =
          gaussian(Di(pixel_p, src_img[thread_id + 1]), lambda, sigma_square);
    }

    if (thread_id + img_width < img_size) {  // down
      edges[idx + 3] = edges[idx + img_width * edges_width + 2] = gaussian(
          Di(pixel_p, src_img[thread_id + img_width]), lambda, sigma_square);
    }
  }
}

__global__ void init(float *res_pixel, float *pixel_flow, int *bin_height,
                     int img_size, int img_height, int img_width, int bin_size) {
  int img_x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x,
      img_y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int img_idx = __umul24(img_y, img_width) + img_x;
  if(img_idx == 0) {
    bin_height[bin_size] = img_size + bin_size + 2;
  }
  if(img_x < img_width && img_y < img_height) {
    float tmp_res = res_pixel[img_idx * RES_UNIT_SIZE + 8];
    if(tmp_res > EPS) {
      pixel_flow[img_idx] = tmp_res;
      res_pixel[img_idx * RES_UNIT_SIZE + 8] = 0;
      res_pixel[img_idx * RES_UNIT_SIZE + 0] += tmp_res;
    }
  }
}


int *getCutMask(int *src_img, int *mask_img, int img_height, int img_width) {
  float lambda = 1.0;
  float beta = 0.5;
  int color_bin_size = 64;
  // int color_bin_num = 256/color_bin_size;
  int color_bin_num = pow(256 / color_bin_size, 3);

  int img_size = img_height * img_width;
  int img_num_bytes = sizeof(int) * img_size;

  // compute sigma square
  int *d_src_img, *d_mask_img;
  cudaMalloc((void **)&d_src_img, img_num_bytes);
  cudaMalloc((void **)&d_mask_img, img_num_bytes);
  cudaMemcpy(d_src_img, src_img, img_num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask_img, mask_img, img_num_bytes, cudaMemcpyHostToDevice);

  computeSigmaSquare(img_height, img_width, d_src_img);

  // compute edges
  float *d_edges = NULL;
  int edges_num_bytes = sizeof(float) * img_num_bytes * (6 + 2 + 2);
  cudaMalloc((void **)&d_edges, edges_num_bytes);

  dim3 block0(1024, 1, 1), grid0(1, 1, 1);
  if (img_size < 1024) {
    block0.x = img_size;
  } else {
    grid0.x = updiv(img_size, 1024);
  }
  computeEdges<<<grid0, block0>>>(lambda, beta, d_edges, img_width, img_height,
                                color_bin_size, d_src_img, d_mask_img);
  CHECK(cudaDeviceSynchronize());

  // initialize data for maxflow
  float *d_bin_flow, *d_pixel_flow, *d_pull_pixel;
  int *d_pixel_height, *d_bin_height;
  bool h_finished, *d_finished;
  float *h_edges = (float *)malloc(edges_num_bytes);
  float *h_pixel_flow = (float *)malloc(img_size * sizeof(float));
  float *h_bin_flow = (float *)malloc((color_bin_num + 1) * sizeof(float));
  int *h_pixel_height = (int *)malloc(img_size * sizeof(int));
  int *h_bin_height = (int *)malloc((color_bin_num + 1) * sizeof(int));

  cudaMalloc((void **)&d_bin_flow, (color_bin_num + 1) * sizeof(float));
  cudaMalloc((void **)&d_pixel_flow, img_size * sizeof(float));
  cudaMalloc((void **)&d_pull_pixel, img_size * sizeof(float));
  cudaMalloc((void **)&d_pixel_height, img_size * sizeof(int));
  cudaMalloc((void **)&d_bin_height, (color_bin_num + 1) * sizeof(int));
  cudaMalloc((void **)&d_finished, sizeof(bool));
  cudaMemcpy(h_edges, d_edges, edges_num_bytes, cudaMemcpyDeviceToHost);
  cudaMemset(d_bin_flow, 0, (color_bin_num + 1) * sizeof(float));
  cudaMemset(d_pixel_flow, 0, img_size * sizeof(float));
  cudaMemset(d_pull_pixel, 0, img_size * sizeof(float));
  cudaMemset(d_pixel_height, 0, img_size * sizeof(int));
  cudaMemset(d_bin_height, 0, (color_bin_num + 1) * sizeof(int));


  dim3 block1(32, 32);
  dim3 grid1(updiv(img_width, 32), updiv(img_height, 32));
  init<<<grid1, block1>>>(d_edges, d_pixel_flow, d_bin_height, img_size, img_height, img_width, color_bin_num);
  // maxflow
  dim3 block_bin(1024);
  dim3 grid_bin(updiv(color_bin_num + 1, 1024));
  do {
    h_finished = true;
    cudaMemcpy(d_finished, &h_finished, sizeof(bool), cudaMemcpyHostToDevice);
    // relabel
    kernel_pixel_relabel<<<grid1, block1,
                           sizeof(int) * (34 * 34 + color_bin_num + 1)>>>(
        d_edges, d_pixel_flow, d_pixel_height, d_bin_height, img_size,
        img_width, img_height, 34 * 34, 34, 34, color_bin_num, d_finished);
    kernel_bin_relabel<<<grid1, block1>>>(
        d_edges, d_pixel_flow, d_bin_flow, d_pixel_height, d_bin_height,
        img_size, img_width, img_height, 34 * 34, 34, 34, color_bin_num,
        d_finished);
    kernel_bin_relabel_rectify<<<grid_bin, block_bin>>>(
        d_bin_height, color_bin_num, d_finished);
    // push & pull
    kernel_pixel_push<<<grid1, block1,
                        34 * 34 * RES_UNIT_SIZE * sizeof(float)>>>(
        d_edges, d_bin_flow, d_pixel_flow, d_pull_pixel, d_pixel_height,
        d_bin_height, img_size, img_width, img_height, 34 * 34, 34, 34,
        color_bin_num);
    kernel_pixel_pull<<<grid1, block1>>>(d_edges, d_pull_pixel, d_pixel_flow,
                                         img_size, img_width, img_height);
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(&h_finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
  } while (!h_finished);

  // bfs
  kernel_bfs_init<<<grid1, block1>>>(d_edges, d_pixel_height, d_bin_height,
                                     img_size, img_width, img_height,
                                     color_bin_num);
  int cur_height = 1;
  do {
    h_finished = true;
    cudaMemcpy(d_finished, &h_finished, sizeof(bool), cudaMemcpyHostToDevice);
    kernel_pixel_bfs<<<grid1, block1,
                       sizeof(int) * (34 * 34 + color_bin_num + 1)>>>(
        d_edges, d_pixel_height, d_bin_height, img_size, img_width, img_height,
        34 * 34, 34, 34, color_bin_num, cur_height, d_finished);
    kernel_bin_bfs<<<grid1, block1, sizeof(int) * color_bin_num>>>(
        d_edges, d_pixel_height, d_bin_height, img_size, img_width, img_height,
        color_bin_num, cur_height, d_finished);
    cudaMemcpy(&h_finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
    cur_height++;
  } while (!h_finished);

  // segment
  kernel_segment<<<grid1, block1>>>(d_pixel_height, img_size, img_width,
                                    img_height);
  cudaMemcpy(h_pixel_height, d_pixel_height, img_size * sizeof(int),
             cudaMemcpyDeviceToHost);

  free(h_edges);
  free(h_bin_flow);
  free(h_bin_height);
  free(h_pixel_flow);

  cudaFree(d_bin_flow);
  cudaFree(d_pixel_flow);
  cudaFree(d_pull_pixel);
  cudaFree(d_pixel_height);
  cudaFree(d_bin_height);
  cudaFree(d_edges);
  cudaFree(d_src_img);
  cudaFree(d_mask_img);

  return h_pixel_height;
}

int main(int argc, char **argv) {
  int img_height, img_width;
//   int img_height = 2, img_width = 3;

  FILE *fp;
  fp = fopen(argv[1], "r");
  fscanf(fp, "%d%d", &img_height, &img_width);

  int *src_img = (int *)malloc(sizeof(int) * img_height * img_width);
  int *mask_img = (int *)malloc(sizeof(int) * img_height * img_width);
  for (int i = 0; i < img_height * img_width; ++i) {
    fscanf(fp, "%d", &src_img[i]);
  }
  for (int i = 0; i < img_height * img_width; ++i) {
    fscanf(fp, "%d", &mask_img[i]);
  }
  fclose(fp);

  int *segment = getCutMask(src_img, mask_img, img_height, img_width);
  for (int j = 0; j < img_width; ++j) {
    for (int i = 0; i < img_height; ++i) {
      printf("%c", segment[i * img_width + j] == 0 ? ' ' : '#');
    }
    printf("\n");
  }
  free(segment);
  return 0;
}
