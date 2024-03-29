﻿/*
 * @Author: X Wang, Y xiao, Ch Yang, G Ye
 * @Date: 2019-06-17 01:05:53
 * @Last Modified by: X Wang, Y Xiao, Ch Yang, G Ye
 * @Last Modified time: 2019-06-17 01:07:15
 * @file description:
    build graph and perform push-relabel
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "build_graph.h"
#include "onecut_kernel.h"
#include "common.h"

__device__ float sigma_square = 0;

/*
  convert to rgb using pixel_value(Int)
*/
__device__ void convertToRGB(int pixel_value, int *r, int *g, int *b) {
  *b = pixel_value & 255;
  pixel_value >>= 8;
  *g = pixel_value & 255;
  pixel_value >>= 8;
  *r = pixel_value & 255;
}

/*
  compute the euclidean distance of p and q pixel
*/
__device__ int Di(int pixel_p, int pixel_q) {
  int p_r, p_g, p_b;
  int q_r, q_g, q_b;
  convertToRGB(pixel_p, &p_r, &p_g, &p_b);
  convertToRGB(pixel_q, &q_r, &q_g, &q_b);
  return (p_r - q_r) * (p_r - q_r) + (p_g - q_g) * (p_g - q_g) +
         (p_b - q_b) * (p_b - q_b);
}

/*
  the unfold warp reduce function
*/
__device__ void warpReduce(volatile int *sigma_sum, int tid, int block_dim_x) {
  sigma_sum[tid] += tid + 32 >= block_dim_x ? 0 : sigma_sum[tid + 32];
  sigma_sum[tid] += tid + 16 >= block_dim_x ? 0 : sigma_sum[tid + 16];
  sigma_sum[tid] += tid + 8 >= block_dim_x ? 0 : sigma_sum[tid + 8];
  sigma_sum[tid] += tid + 4 >= block_dim_x ? 0 : sigma_sum[tid + 4];
  sigma_sum[tid] += tid + 2 >= block_dim_x ? 0 : sigma_sum[tid + 2];
  sigma_sum[tid] += tid + 1 >= block_dim_x ? 0 : sigma_sum[tid + 1];
}

/*
  compute the sigma square sum before building graph;
  each thread computes two pixels' euclidean distance of 4-neighbors
*/
__global__ void computeSigmaSquareSum(int img_width, int img_height,
                                      const int *__restrict__ src_img) {
  extern __shared__ int sigma_sum[];

  // the id of thread in this block
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
  // use reduction for suming up the sigma_sum
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sigma_sum[tid] += sigma_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warpReduce(sigma_sum, tid, blockDim.x);
  }

  // the first thread of each block write result into global variate
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

  cudaMemcpyToSymbol(sigma_square, (void *)&h_sigma_square, sizeof(float));
}

__device__ float gaussian(int di, float lambda, float sigma_square) {
  return lambda * exp(-di / (2 * sigma_square));
}

/*
  compute the bin index of given pixel_value
*/
__device__ int getColorBinIdx(int pixel_value, int color_bin_size) {
  int r, g, b;
  convertToRGB(pixel_value, &r, &g, &b);

  int per_bin_channel = 256 / color_bin_size;
  return (r / color_bin_size) * per_bin_channel * per_bin_channel +
         (g / color_bin_size) * per_bin_channel + (b / color_bin_size);
}

/*
  compute the edges of given src_img and mask_img;
  each thread compute for one pixel point;
*/
__global__ void computeEdges(float lambda, float beta, unsigned int *edges,
                             int img_width, int img_height, int color_bin_size,
                             int *bin_idx,
                             const int *__restrict__ src_img,
                             const int *__restrict__ mask_img) {

  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;

  int img_size = img_height * img_width;
  int edges_width = 6 + 2 + 2;

  if (thread_id < img_size) {
    int idx = thread_id * (edges_width);

    // add s-t-links or t-t-links
    int seed_value = mask_img[thread_id];
    if (seed_value == 255 << 16) {  // s-t-links
      edges[idx] = edges[idx + 8] = MAX;
    } else if (seed_value == 255 << 8) {  // t-t-links
      edges[idx + 1] = MAX;
    }

    // add a-link of color bins
    int color_bin_idx = getColorBinIdx(src_img[thread_id], color_bin_size);
    bin_idx[color_bin_idx] = 1;
    edges[idx + 5 + 1] = color_bin_idx;
    edges[idx + 5 + 2] = edges[idx + 9] = beta * coefficient;

    // add n-links
    int pixel_p = src_img[thread_id];
    if (thread_id % img_width + 1 < img_width) {  // right
      edges[idx + 5] = edges[idx + edges_width + 4] =
          coefficient *
          gaussian(Di(pixel_p, src_img[thread_id + 1]), lambda, sigma_square);
    }

    if (thread_id + img_width < img_size) {  // down
      edges[idx + 3] = edges[idx + img_width * edges_width + 2] =
          coefficient * gaussian(Di(pixel_p, src_img[thread_id + img_width]),
                                 lambda, sigma_square);
    }
  }
}

__global__ void init(unsigned int *res_pixel, unsigned int *pixel_flow,
                     int *bin_height, int *height_count, int img_size,
                     int img_height, int img_width, int bin_size) {
  int img_x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x,
      img_y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int img_idx = __umul24(img_y, img_width) + img_x;
  if (img_idx == 0) {
    bin_height[bin_size] = img_size + bin_size + 2;
    height_count[0] = img_size + bin_size + 1;
    height_count[img_size + bin_size + 2] = 1;
  }
  if (img_x < img_width && img_y < img_height) {
    unsigned int tmp_res = res_pixel[img_idx * RES_UNIT_SIZE + 8];
    if (tmp_res > 0) {
      pixel_flow[img_idx] = tmp_res;
      res_pixel[img_idx * RES_UNIT_SIZE + 8] = 0;
      res_pixel[img_idx * RES_UNIT_SIZE + 0] += tmp_res;
    }
  }
}

/*
  update the bin index after compressing bins
*/
__global__ void updateBinIdx(int img_height, int img_width,
                             unsigned int *edges, 
                             const int *__restrict__ bin_idx) {
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;
  
  int img_size = img_height * img_width;
  int edges_width = 6 + 2 + 2;

  if (thread_id < img_size) {
    int idx = thread_id * (edges_width);
    edges[idx + 5 + 1] = bin_idx[edges[idx + 5 + 1]];
  }
}

/*
  the host function of build graph;
  Return:: 
    @unsigned int array: the edges of the graph
*/
unsigned int *buildGraph(int *src_img, int *mask_img, 
                         int img_height, int img_width, int *ptr_color_bin_num) {
  int img_size = img_height * img_width;
  int img_num_bytes = sizeof(unsigned int) * img_size;

  // compute sigma square
  int *d_src_img, *d_mask_img;
  cudaMalloc((void **)&d_src_img, img_num_bytes);
  cudaMalloc((void **)&d_mask_img, img_num_bytes);
  cudaMemcpy(d_src_img, src_img, img_num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask_img, mask_img, img_num_bytes, cudaMemcpyHostToDevice);

  computeSigmaSquare(img_height, img_width, d_src_img);

  // compute edges
  unsigned int *d_edges = NULL;
  int edges_num_bytes = sizeof(int) * img_size * (6 + 2 + 2);
  cudaMalloc((void **)&d_edges, edges_num_bytes);
  cudaMemset(d_edges, 0, edges_num_bytes);

  int *d_bin_idx = NULL;
  int bin_num_bytes = sizeof(int) * (*ptr_color_bin_num);
  cudaMalloc((void**)&d_bin_idx, bin_num_bytes);
  cudaMemset(d_bin_idx, 0, bin_num_bytes);

  dim3 block0(1024, 1, 1), grid0(1, 1, 1);
  if (img_size < 1024) {
    block0.x = img_size;
  } else {
    grid0.x = updiv(img_size, 1024);
  }
  computeEdges<<<grid0, block0>>>(lambda, beta, d_edges, img_width, img_height,
                                  color_bin_size, d_bin_idx, d_src_img, 
                                  d_mask_img);

  int *h_bin_idx = (int*)malloc(bin_num_bytes);
  cudaMemcpy(h_bin_idx, d_bin_idx, bin_num_bytes, cudaMemcpyDeviceToHost);
  
  // compress the number of bins
  int idx = 0;
  for (int i = 0; i < (*ptr_color_bin_num); ++i) {
    if (h_bin_idx[i]) {
      h_bin_idx[i] = idx++;
    }
  }
  *ptr_color_bin_num = idx;

  cudaMemcpy(d_bin_idx, h_bin_idx, bin_num_bytes, cudaMemcpyHostToDevice);
  updateBinIdx<<<grid0, block0>>>(img_height, img_width, d_edges, d_bin_idx);
  CHECK(cudaDeviceSynchronize());

  free(h_bin_idx);
    
  cudaFree(d_bin_idx);
  cudaFree(d_src_img);
  cudaFree(d_mask_img);

  return d_edges;
}

int *maxFlow(int img_height, int img_width, unsigned int *d_edges,
             int color_bin_num) {
  int img_size = img_height * img_width;
  int max_height = (img_size + color_bin_num + 2) * 2;

  // initialize data for maxflow
  unsigned long long *d_bin_flow;
  unsigned int *d_pixel_flow, *d_pull_pixel;
  int *d_pixel_height, *d_bin_height, *d_new_bin_height;
  int *d_height_count, *h_gap, *d_gap;
  bool *h_finished, *d_finished;
  int *h_pixel_height = (int *)malloc(img_size * sizeof(int));
  int *h_bin_height = (int *)malloc((color_bin_num + 1) * sizeof(int));
  cudaMallocHost((void **)&h_finished, sizeof(bool));
  cudaMallocHost((void **)&h_gap, sizeof(int));

  cudaMalloc((void **)&d_bin_flow,
             (color_bin_num + 1) * sizeof(unsigned long long));
  cudaMalloc((void **)&d_pixel_flow, img_size * sizeof(unsigned int));
  cudaMalloc((void **)&d_pull_pixel, img_size * sizeof(unsigned int));
  cudaMalloc((void **)&d_pixel_height, img_size * sizeof(int));
  cudaMalloc((void **)&d_bin_height, (color_bin_num + 1) * sizeof(int));
  cudaMalloc((void **)&d_new_bin_height, (color_bin_num + 1) * sizeof(int));
  cudaMalloc((void **)&d_height_count, max_height * sizeof(int));
  cudaMalloc((void **)&d_gap, sizeof(int));
  cudaMalloc((void **)&d_finished, sizeof(bool));
  cudaMemset(d_bin_flow, 0, (color_bin_num + 1) * sizeof(unsigned long long));
  cudaMemset(d_pixel_flow, 0, img_size * sizeof(unsigned int));
  cudaMemset(d_pull_pixel, 0, img_size * sizeof(unsigned int));
  cudaMemset(d_pixel_height, 0, img_size * sizeof(int));
  cudaMemset(d_bin_height, 0, (color_bin_num + 1) * sizeof(int));
  cudaMemset(d_height_count, 0, max_height * sizeof(int));

  dim3 block_pixel(32, 32);
  dim3 grid_pixel(updiv(img_width, 32), updiv(img_height, 32));
  init<<<grid_pixel, block_pixel>>>(d_edges, d_pixel_flow, d_bin_height,
                                    d_height_count, img_size, img_height,
                                    img_width, color_bin_num);
  cudaMemcpy(d_new_bin_height, d_bin_height, color_bin_num * sizeof(int),
             cudaMemcpyDeviceToDevice);
  // maxflow
  dim3 block_bin(1024);
  dim3 grid_bin(updiv(color_bin_num + 1, 1024));
  dim3 block_height(1024);
  dim3 grid_height(updiv(max_height, 1024));
  do {
    *h_finished = true;
    cudaMemcpy(d_finished, h_finished, sizeof(bool), cudaMemcpyHostToDevice);
    // find gap
    *h_gap = INF;
    cudaMemcpy(d_gap, h_gap, sizeof(int), cudaMemcpyHostToDevice);
    kernel_check_gap<<<grid_height, block_height>>>(
        d_height_count, d_gap, img_size + color_bin_num + 2);
    cudaMemcpy(h_gap, d_gap, sizeof(int), cudaMemcpyDeviceToHost);
    // relabel
    kernel_gap_relabel<<<grid_height, block_height>>>(
        d_pixel_height, d_bin_height, d_height_count, img_size, color_bin_num,
        *h_gap);
    kernel_pixel_relabel<<<grid_pixel, block_pixel, sizeof(int) * (34 * 34)>>>(
        d_edges, d_pixel_flow, d_pixel_height, d_bin_height, d_height_count,
        d_gap, img_size, img_width, img_height, 34 * 34, 34, 34, color_bin_num,
        d_finished);
    kernel_bin_relabel<<<grid_pixel, block_pixel>>>(
        d_edges, d_pixel_flow, d_bin_flow, d_pixel_height, d_bin_height,
        d_new_bin_height, img_size, img_width, img_height, 34 * 34, 34, 34,
        color_bin_num);
    kernel_bin_relabel_update<<<grid_bin, block_bin>>>(
        d_bin_height, d_new_bin_height, d_height_count, color_bin_num,
        max_height, d_finished);
    // push & pull
    kernel_pixel_push<<<grid_pixel, block_pixel,
                        34 * 34 * RES_UNIT_SIZE * sizeof(unsigned int)>>>(
        d_edges, d_bin_flow, d_pixel_flow, d_pull_pixel, d_pixel_height,
        d_bin_height, img_size, img_width, img_height, 34 * 34, 34, 34,
        color_bin_num);
    kernel_pixel_pull<<<grid_pixel, block_pixel>>>(
        d_edges, d_pull_pixel, d_pixel_flow, img_size, img_width, img_height);
    cudaMemcpy(h_finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
  } while (!(*h_finished));

  // bfs
  kernel_bfs_init<<<grid_pixel, block_pixel>>>(
      d_edges, d_pixel_height, d_bin_height, img_size, img_width, img_height,
      color_bin_num);
  int cur_height = 1;
  do {
    *h_finished = true;
    cudaMemcpy(d_finished, h_finished, sizeof(bool), cudaMemcpyHostToDevice);
    kernel_pixel_bfs<<<grid_pixel, block_pixel, sizeof(int) * (34 * 34)>>>(
        d_edges, d_pixel_height, d_bin_height, img_size, img_width, img_height,
        34 * 34, 34, 34, color_bin_num, cur_height, d_finished);
    kernel_bin_bfs<<<grid_pixel, block_pixel>>>(
        d_edges, d_pixel_height, d_bin_height, img_size, img_width, img_height,
        color_bin_num, cur_height, d_finished);
    cudaMemcpy(h_finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
    cur_height++;
  } while (!(*h_finished));

  // segment
  kernel_segment<<<grid_pixel, block_pixel>>>(d_pixel_height, img_size,
                                              img_width, img_height);
  cudaMemcpy(h_pixel_height, d_pixel_height, img_size * sizeof(int),
             cudaMemcpyDeviceToHost);

  free(h_bin_height);
  cudaFreeHost(h_gap);
  cudaFreeHost(h_finished);

  cudaFree(d_finished);
  cudaFree(d_bin_flow);
  cudaFree(d_pixel_flow);
  cudaFree(d_pull_pixel);
  cudaFree(d_pixel_height);
  cudaFree(d_bin_height);
  cudaFree(d_height_count);
  cudaFree(d_gap);

  return h_pixel_height;
}

/*
  the host function of getCutMask using onecut;
  Return::
    @Int array: the result of segment;
                0 for foreground and 255 for background
*/
int *getCutMask(int *src_img, int *mask_img, int img_height, int img_width) {
  int color_bin_num = pow(256 / color_bin_size, 3);
  unsigned int *d_edges =
      buildGraph(src_img, mask_img, img_height, img_width, &color_bin_num);

  int *segment = maxFlow(img_height, img_width, d_edges, color_bin_num);

  cudaFree(d_edges);

  return segment;
}


int *maxFlow_iSAP(int img_height, int img_width,
             unsigned int *d_edges, int color_bin_num) {

  int img_size = img_height * img_width;
  int edges_num_bytes = sizeof(int) * img_size * (6 + 2 + 2);

  unsigned int *capacity = (unsigned int *)malloc(edges_num_bytes);
  cudaMemcpy(capacity, d_edges, edges_num_bytes, cudaMemcpyDeviceToHost);
  int *mask = (int *)malloc(img_size * sizeof(int));

  serialMaxflow(capacity, img_size, img_width, img_height, color_bin_num, mask);

  free(capacity);

  return mask;
}

int *getCutMask_iSAP(int *src_img, int *mask_img, int img_height, int img_width) {
  int color_bin_num = pow(256 / color_bin_size, 3);
  unsigned int *d_edges = buildGraph(src_img, mask_img,
                                     img_height, img_width, &color_bin_num);

  int *segment = maxFlow(img_height, img_width, d_edges, color_bin_num);

  cudaFree(d_edges);

  return segment;
}

void serialMaxflow(unsigned *res, int img_size, int col, int row, int bin_num, int *mask) {
  /*
  Find maxflow by iSAP and generate mask.
  */
  // convert the graph representation
  int n = img_size + bin_num + 2;
  int S = img_size + bin_num;
  int T = S + 1;
  long long ans = 0;
  struct Edge {
    int lh, from, to;
    int f, c;
    Edge() {}
    Edge(int lh, int from, int to, int f, int c): lh(lh), from(from), to(to), f(f), c(c) {}
  };
  std::vector<Edge> edges;
  std::vector<int> last_edge(n + 2, -1);
  for (int i = 0; i < img_size; i++) {
    unsigned *res_pixel = res + i * RES_UNIT_SIZE;
    int x = i / col, y = i % col;
    if (x > 0 && res_pixel[2] > 0) {   // up-down
      edges.push_back(Edge(last_edge[i], i, i - col, 0, res_pixel[2]));
      last_edge[i] = (int)edges.size() - 1;
      edges.push_back(Edge(last_edge[i - col], i - col, i, 0, res_pixel[2]));
      last_edge[i - col] = (int)edges.size() - 1;
    }
    if (y > 0 && res_pixel[4] > 0) { // left-right
      edges.push_back(Edge(last_edge[i], i, i - 1, 0, res_pixel[4]));
      last_edge[i] = (int)edges.size() - 1;
      edges.push_back(Edge(last_edge[i - 1], i - 1, i, 0, res_pixel[4]));
      last_edge[i - 1] = (int)edges.size() - 1;
    }
    if (res_pixel[8] > 0) { // S-pixel
      edges.push_back(Edge(last_edge[S], S, i, 0, res_pixel[8]));
      last_edge[S] = (int)edges.size() - 1;
      edges.push_back(Edge(last_edge[i], i, S, 0, res_pixel[8]));
      last_edge[i] = (int)edges.size() - 1;
    }
    if (res_pixel[1] > 0) { // pixel-T
      edges.push_back(Edge(last_edge[i], i, T, 0, res_pixel[1]));
      last_edge[i] = (int)edges.size() - 1;
      edges.push_back(Edge(last_edge[T], T, i, 0, res_pixel[1]));
      last_edge[T] = (int)edges.size() - 1;
    }
    if (res_pixel[9] > 0) { // pixel-bin
      int bid = res_pixel[6] + img_size;
      edges.push_back(Edge(last_edge[i], i, bid, 0, res_pixel[9]));
      last_edge[i] = (int)edges.size() - 1;
      edges.push_back(Edge(last_edge[bid], bid, i, 0, res_pixel[9]));
      last_edge[bid] = (int)edges.size() - 1;
    }
  }
  // iSAP
  std::vector<int> cur(last_edge), pre(n + 2, 0), d(n + 2, 0), gap(n + 2, 0);
  int p, i, k;
  p = S;
  pre[S] = S;
  gap[0] = n;
  while (d[S] <= n) {
    for (i = cur[p]; i >= 0; i = edges[i].lh) {   // push
      if (d[p] == d[edges[i].to] + 1 && edges[i].f < edges[i].c) {
        cur[p] = i;
        k = edges[i].to;
        pre[k] = p;
        p = k;
        break;
      }
    }
    if (i == -1) {  // relabel
      if (gap[d[p]] == 1) break;
      gap[d[p]]--;
      cur[p] = last_edge[p];
      k = n;
      for (i = last_edge[p]; i >= 0; i = edges[i].lh)
        if (edges[i].f < edges[i].c && d[edges[i].to] < k) k = d[edges[i].to];
      d[p] = k + 1;
      gap[k + 1]++;
      p = pre[p];
    }
    if (p == T) { // find an augmented path
      k = edges[cur[S]].c - edges[cur[S]].f;
      for (i = pre[p]; i != S; i = pre[i])
        if (edges[cur[i]].c - edges[cur[i]].f < k)
          k = edges[cur[i]].c - edges[cur[i]].f;
      for (i = pre[p]; i != S; i = pre[i]) {
        edges[cur[i]].f += k;
        edges[cur[i] ^ 1].f -= k;
      }
      edges[cur[S]].f += k;
      edges[cur[S] ^ 1].f -= k;
      ans += k;
      p = S;
    }
  }
  int *queue = (int *)malloc(sizeof(int) * (img_size + bin_num));
  int *height = (int *)malloc(sizeof(int) * (img_size + bin_num));
  int head = 0, tail = 0;
  for (i = 0; i < img_size + bin_num; i++) {
    height[i] = 1000000000;
  }
  for (i = last_edge[S]; i >= 0; i = edges[i].lh) {
    if (edges[i].f > 0) {
      height[edges[i].to] = 1;
      queue[tail++] = edges[i].to;
    }
  }
  while (head < tail) {
    p = queue[head++];
    int new_height = height[p] + 1;
    for (i = last_edge[p]; i >= 0; i = edges[i].lh) {
      int q = edges[i].to;
      if (edges[i].f < edges[i].c && height[q] > new_height) {
        height[q] = new_height;
        queue[tail++] = q;
      }
    }
  }
  for (i = 0; i < img_size; i++) {
    mask[i] = (height[i] < 1000000000) ? 255 : 0;
  }
}
