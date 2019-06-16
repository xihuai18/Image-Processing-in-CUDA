/*
 * @Author: X Wang, Y xiao, Ch Yang, G Ye 
 * @Date: 2019-06-17 00:38:46 
 * @Last Modified by: X Wang, Y xiao, Ch Yang, G Ye
 * @Last Modified time: 2019-06-17 00:40:08
 * @file description:
    functions for push-relabel algorithm, including push, pull, relabel and the function to segment the image
*/


#ifndef __ONECUT__KERNEL__
#define __ONECUT__KERNEL__ value

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdio.h>

#define EPS 1e-6
#define RES_UNIT_SIZE 10
#define INF 1000000000

/*
Variables:
  res_pixel: (6+2+2) * pixel_num , unsigned int
    reserved flow from and to pixel
  flow_bin: bin_num, unsigned long long
    the left flow in bin, 0
  flow_pixel: pixel_num, unsigned int
    the left flow in pixel, 0
  pull_pixel: pixel_num, unsigned int
    flow pulled from other pixels or bins/source, 0
  pixel_height: pixel_num, int
    height of pixel, 0
  bin_height: bin_num, int
    height of pixel, 0
  finished: 1, bool
    whether the algorithm iteration is finished, true
*/

// pixel-push->pull_pixel,bin
// shared memory: blockDim.x * blockDim.y * RES_UNIT_SIZE * sizeof(unsigned int)
__global__ void kernel_pixel_push(unsigned int *res_pixel,
                                  unsigned long long *bin_flow,
                                  unsigned int *pixel_flow,
                                  unsigned int *pull_pixel, int *pixel_height,
                                  int *bin_height, int img_size, int col,
                                  int row, int tile_size, int tile_col,
                                  int tile_row, int bin_num);

// pixel<-pull-pull_pixel
__global__ void kernel_pixel_pull(unsigned int *res_pixel,
                                  unsigned int *pull_pixel,
                                  unsigned int *pixel_flow, int img_size,
                                  int col, int row);

// relabel pixel height
__global__ void kernel_pixel_relabel(unsigned int *res,
                                     unsigned int *pixel_flow,
                                     int *pixel_height, int *bin_height,
                                     int *height_count, int *gap, int img_size,
                                     int col, int row, int tile_size,
                                     int tile_col, int tile_row, int bin_num,
                                     bool *finished);

// relabel bin by pixel
__global__ void kernel_bin_relabel(unsigned int *res, unsigned int *pixel_flow,
                                   unsigned long long *bin_flow,
                                   int *pixel_height, int *bin_height,
                                   int *new_bin_height, int img_size, int col,
                                   int row, int tile_size, int tile_col,
                                   int tile_row, int bin_num);

// must be called after kernel_bin_relabel
__global__ void kernel_bin_relabel_update(int *bin_height, int *new_bin_height,
                                          int *height_count, int bin_num,
                                          int max_height, bool *finished);

// find the gap
__global__ void kernel_check_gap(int *height_count, int *gap, int num);

// relabel pixels & bins whose height is in (gap, n) to n + 1
__global__ void kernel_gap_relabel(int *pixel_height, int *bin_height,
                                   int *height_count, int img_size, int bin_num,
                                   int gap);

// init the pixels connected to source and the pixels connected to sink, 1, -1
// or MAX
__global__ void kernel_bfs_init(unsigned int *res, int *bfs_pixel_height,
                                int *bfs_bin_height, int img_size, int col,
                                int row, int bin_num);

// pixel assign cur_height + 1 to vaild pixel and bin
// the bfs_bin_height and bfs_pixel_height are initialized to MAX
// Size of shared memory: 4 * (tile_size + bin_num + 1)
__global__ void kernel_pixel_bfs(unsigned int *res, int *bfs_pixel_height,
                                 int *bfs_bin_height, int img_size, int col,
                                 int row, int tile_size, int tile_col,
                                 int tile_row, int bin_num, int cur_height,
                                 bool *finished);

// pixel change height to cur_height + 1 if it is connected from bin with
// height = cur_height
// Size of shared memory: 4 * bin_num
__global__ void kernel_bin_bfs(unsigned int *res, int *bfs_pixel_height,
                               int *bfs_bin_height, int img_size, int col,
                               int row, int bin_num, int cur_height,
                               bool *finished);

// segment the image into 255 or 0
__global__ void kernel_segment(int *bfs_pixel_heiht, int img_size, int col,
                               int row);

#endif
