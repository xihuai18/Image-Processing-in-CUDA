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

// pixel -> pixel/bin/source/sink, bin -> pixel
// Size of shared memory: sizeof(unsigned) * tile_size * RES_UNIT_SIZE
__global__ void kernel_pixel_push(unsigned int* res_pixel,
                                  unsigned long long* bin_flow,
                                  unsigned int* pixel_flow,
                                  unsigned int* pull_pixel, int* pixel_height,
                                  int* bin_height, int img_size, int col,
                                  int row, int tile_size, int tile_col,
                                  int tile_row, int bin_num);

// update pixel_flow by pull_pixel
__global__ void kernel_pixel_pull(unsigned int* res_pixel,
                                  unsigned int* pull_pixel,
                                  unsigned int* pixel_flow, int img_size,
                                  int col, int row);

// relabel pixels
// Size of shared memory: sizeof(int) * tile_size
__global__ void kernel_pixel_relabel(unsigned int* res,
                                     unsigned int* pixel_flow,
                                     int* pixel_height, int* bin_height,
                                     int* height_count, int* gap, int img_size,
                                     int col, int row, int tile_size,
                                     int tile_col, int tile_row, int bin_num,
                                     bool* finished);

// relabel bins
__global__ void kernel_bin_relabel(unsigned int* res, unsigned int* pixel_flow,
                                   unsigned long long* bin_flow,
                                   int* pixel_height, int* bin_height,
                                   int* new_bin_height, int img_size, int col,
                                   int row, int tile_size, int tile_col,
                                   int tile_row, int bin_num);

// must be called after kernel_bin_relabel
__global__ void kernel_bin_relabel_update(int* bin_height, int* new_bin_height,
                                           int* height_count, int bin_num,
                                           int max_height, bool* finished);

// find the gap
__global__ void kernel_check_gap(int* height_count, int* gap, int source_height);

// relabel pixels/bins whose height is between gap and source height
__global__ void kernel_gap_relabel(int* pixel_height, int* bin_height,
                                   int* height_count, int img_size, int bin_num,
                                   int gap);

// Initialization for BFS
__global__ void kernel_bfs_init(unsigned int* res, int* bfs_pixel_height,
                                int* bfs_bin_height, int img_size, int col,
                                int row, int bin_num);

// BFS expansion for pixels
// Size of shared memory: sizeof(int) * tile_size
__global__ void kernel_pixel_bfs(unsigned int* res, int* bfs_pixel_height,
                                 int* bfs_bin_height, int img_size, int col,
                                 int row, int tile_size, int tile_col,
                                 int tile_row, int bin_num, int cur_height,
                                 bool* finished);

// BFS expansion for bins
__global__ void kernel_bin_bfs(unsigned int* res, int* bfs_pixel_height,
                               int* bfs_bin_height, int img_size, int col,
                               int row, int bin_num, int cur_height,
                               bool* finished);

// generate segmentation mask (0/255)
__global__ void kernel_segment(int* bfs_pixel_heiht, int img_size, int col,
                               int row);

#endif
