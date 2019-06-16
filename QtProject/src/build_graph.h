#ifndef BUILD_GRAPH_H
#define BUILD_GRAPH_H

#include <cuda.h>
#include <cuda_runtime.h>

const float lambda = 1.0;
const float beta = 0.5;
const int color_bin_size = 8;
const int coefficient = 1e8;
const unsigned int MAX = 1000000000;

__device__ void convertToRGB(int pixel_value, int *r, int *g, int *b);

__device__ int Di(int pixel_p, int pixel_q);

__device__ void warpReduce(volatile int *sigma_sum, int tid, int block_dim_x);

__global__ void computeSigmaSquareSum(int img_width, int img_height,
                                      const int *__restrict__ src_img);

__host__ void computeSigmaSquare(int img_height, int img_width,
                                 int *d_src_img);

__device__ float gaussian(int di, float lambda, float sigma_square);

__device__ int getColorBinIdx(int pixel_value, int color_bin_size);

__global__ void computeEdges(float lambda, float beta, float *edges,
                             int img_width, int img_height, int color_bin_size,
                             int *bin_idx,
                             const int *__restrict__ src_img,
                             const int *__restrict__ mask_img);

__global__ void init(unsigned int *res_pixel, unsigned int *pixel_flow,
                     int *bin_height, int img_size, int img_height,
                     int img_width, int bin_size);

__global__ void updateBinIdx(int img_height, int img_width,
                             unsigned int *edges,
                             const int *__restrict__ bin_idx);


unsigned int* buildGraph(int *src_img, int *mask_img,
                         int img_height, int img_width, int *ptr_color_bin_num);

int* maxFlow(int img_height, int img_width,
             unsigned int *d_edges, int color_bin_num);
int *maxFlow_iSAP(int img_height, int img_width,
             unsigned int *d_edges, int color_bin_num);
void serialMaxflow(unsigned *res, int img_size, int col, int row, int bin_num, int *mask);
extern "C"
int* getCutMask(int *src_img, int *mask_img, int img_height, int img_width);
extern "C"
int* getCutMask_iSAP(int *src_img, int *mask_img, int img_height, int img_width);

#endif
