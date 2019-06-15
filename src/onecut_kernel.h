#ifndef __ONECUT__KERNEL__
#define __ONECUT__KERNEL__ value

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdio.h>

#define EPS 1e-6
#define RES_UNIT_SIZE 10
#define INF 1000000000

#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess) {                              \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
      exit(1);                                               \
    }                                                        \
  }

// __device__ bool fgreater(float x, float y);

// __device__ bool fless(float x, float y);

// __device__ bool fequal(float x, float y);

// __device__ static float atomicCAS(float* address, float compared, float val);

__global__ void kernel_pixel_push(unsigned int* res_pixel, unsigned long long* bin_flow,
                                  unsigned int* pixel_flow, unsigned int* pull_pixel,
                                  int* pixel_height, int* bin_height,
                                  int img_size, int col, int row, int tile_size,
                                  int tile_col, int tile_row, int bin_num);
// pixel-push->pull_pixel,bin

__global__ void kernel_pixel_pull(unsigned int* res_pixel, unsigned int* pull_pixel,
                                  unsigned int* pixel_flow, int img_size, int col,
                                  int row);
// pixel<-pull-pull_pixel

__global__ void kernel_pixel_relabel(unsigned int* res,
                                     unsigned int* pixel_flow,
                                     int* pixel_height, int* bin_height,
                                     int img_size, int col, int row,
                                     int tile_size, int tile_col, int tile_row,
                                     int bin_num, bool* finished);
// relabel pixel height
// Size of shared memory: 4 * (tile_size + bin_num + 1)

__global__ void kernel_bin_relabel(unsigned int* res, unsigned int* pixel_flow,
                                   unsigned long long* bin_flow,
                                   int* pixel_height, int* bin_height,
                                   int img_size, int col, int row,
                                   int tile_size, int tile_col, int tile_row,
                                   int bin_num, bool* finished);
// relabel bin by pixel
// atomicCAS();

__global__ void kernel_bin_relabel_rectify(int* bin_height, int bin_num,
    bool* finished);
// must be called after kernel_bin_relabel

__global__ void kernel_bfs_init(unsigned int* res, int* bfs_pixel_height,
                                int* bfs_bin_height, int img_size, int col,
                                int row, int bin_num);
// init the pixels connected to source and the pixels connected to sink, 1, -1
// or MAX

__global__ void kernel_pixel_bfs(unsigned int* res, int* bfs_pixel_height,
                                 int* bfs_bin_height, int img_size, int col,
                                 int row, int tile_size, int tile_col,
                                 int tile_row, int bin_num, int cur_height,
                                 bool* finished);
// pixel assign cur_height + 1 to vaild pixel and bin
// the bfs_bin_height and bfs_pixel_height are initialized to MAX
// Size of shared memory: 4 * (tile_size + bin_num + 1)

__global__ void kernel_bin_bfs(unsigned int* res, int* bfs_pixel_height,
                               int* bfs_bin_height, int img_size, int col,
                               int row, int bin_num, int cur_height,
                               bool* finished);
// pixel change height to cur_height + 1 if it is connected from bin with
// height = cur_height
// Size of shared memory: 4 * bin_num

__global__ void kernel_segment(int* bfs_pixel_heiht, int img_size, int col,
                               int row);
// 255 or 0

#endif
