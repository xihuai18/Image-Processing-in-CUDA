/*
 * @Author: X Wang, Y xiao, Ch Yang, G Ye
 * @Date: 2019-06-17 00:56:33
 * @Last Modified by: X Wang, Y Xiao, Ch Yang, G Ye
 * @Last Modified time: 2019-06-17 00:57:38
 * @file description:
    Contrast Local Adaptive Histgram Enhancement
 */
#include "common.h"
#include "enhance.h"
#include <stdio.h>
#include <stdlib.h>


/*
Contrast Local Adaptive Histgram Enhancement 
*/
__global__ void CLAHE(int * hsi_img, int height, int width)
// the 'tile' size is the same with the block size, 1 block for 9 tile
{
    __shared__ int frq[9*256+9];
    int lt_x = __umul24(blockIdx.x, blockDim.x*3) + threadIdx.x,
      lt_y = __umul24(blockIdx.y, blockDim.y*3) + threadIdx.y;
    int lt_idx = __umul24(lt_y, width) + lt_x;
    int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
    int per_thread = 9;
    if(thread_idx < 256) {
        for (int i = 0; i < per_thread; ++i)
        {
            frq[thread_idx*per_thread+i] = 0;
        }
    }
    if (thread_idx == 0) {
        // printf("%d %d\n", blockIdx.x, blockIdx.y);
        for (int i = 0; i < 9; ++i)
        {
            frq[9*256+i] = 0;
        }
    }

    __syncthreads();

    for (int i = 0; i < 3; ++i)
    {
        int tmp_x = lt_x;
        int tmp_y = lt_y + i*TILESIZE;
        int tmp_idx = __umul24(tmp_y, width) + tmp_x;
        if(tmp_x < width && tmp_y < height) {
            atomicAdd(&frq[(i*3+0)*256+(hsi_img[tmp_idx]&0x0000FF)], 1);
        }
        tmp_x = lt_x + TILESIZE;
        tmp_idx = __umul24(tmp_y, width) + tmp_x;
        if(tmp_x < width && tmp_y < height) {
            atomicAdd(&frq[(i*3+1)*256+(hsi_img[tmp_idx]&0x0000FF)], 1);
        }
        tmp_x = lt_x + TILESIZE*2;
        tmp_idx = __umul24(tmp_y, width) + tmp_x;
        if(tmp_x < width && tmp_y < height) {
            atomicAdd(&frq[(i*3+2)*256+(hsi_img[tmp_idx]&0x0000FF)], 1);
        }
    }
    __syncthreads();

    if(thread_idx < 256) {
        for (int i = 0; i < 9; ++i)
        {
            int overflow = (frq[i*256+thread_idx] > THRESHOLD)? frq[i*256+thread_idx] - THRESHOLD : 0;
            frq[i*256+thread_idx] -= overflow;
            atomicAdd(&frq[9*256+i], overflow);
        }
    }
    __syncthreads();

    if(thread_idx < 256) {
        for (int i = 0; i < 9; ++i)
        {
            frq[i*256+thread_idx] += frq[9*256+i]/256;
        }
    }

    __syncthreads();

    for (int i = 0; i < 9; ++i)
    {
        for (int stride = 1; stride < 256; stride <<= 1)
        {
            __syncthreads();
            int val;
            if(thread_idx < 256)
                val = (thread_idx > stride)? frq[i*256+thread_idx-stride]:0;
            __syncthreads();
            if(thread_idx < 256)
                frq[i*256+thread_idx] += val;
        }
    }

    __syncthreads();

    for (int i = 0; i < 3; ++i)
    {
        int tmp_x = lt_x;
        int tmp_y = lt_y + i*TILESIZE;
        int tmp_idx = __umul24(tmp_y, width) + tmp_x;
        if(tmp_x < width && tmp_y < height) {
            hsi_img[tmp_idx] = (hsi_img[tmp_idx] & 0xFFFF00) + (1.0*frq[(i*3+0)*256+(hsi_img[tmp_idx]&0x0000FF)]/(TILESIZE*TILESIZE))*255;
        }
        tmp_x = lt_x + TILESIZE;
        tmp_idx = __umul24(tmp_y, width) + tmp_x;
        if(tmp_x < width && tmp_y < height) {
            hsi_img[tmp_idx] = (hsi_img[tmp_idx] & 0xFFFF00) + (1.0*frq[(i*3+1)*256+(hsi_img[tmp_idx]&0x0000FF)]/(TILESIZE*TILESIZE))*255;
        }
        tmp_x = lt_x + TILESIZE*2;
        tmp_idx = __umul24(tmp_y, width) + tmp_x;
        if(tmp_x < width && tmp_y < height) {
            hsi_img[tmp_idx] = (hsi_img[tmp_idx] & 0xFFFF00) + (1.0*frq[(i*3+2)*256+(hsi_img[tmp_idx]&0x0000FF)]/(TILESIZE*TILESIZE))*255;
        }
    }
}

int* imgCLAHE(int *src_img, int img_height, int img_width)
{
    int * d_rgb_img, * d_hsi_img;
    int * ret_img;
    int *h_img_one, *h_img_two;
    ret_img = (int*)malloc(img_height*img_width*sizeof(int));
    h_img_one = (int*)malloc(img_height*img_width*sizeof(int));
    h_img_two = (int*)malloc(img_height*img_width*sizeof(int));
    cudaMalloc((void**)& d_rgb_img, img_height*img_width*sizeof(int));
    cudaMalloc((void**)& d_hsi_img, img_height*img_width*sizeof(int));
    cudaMemcpy(d_rgb_img, src_img, img_height*img_width*sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(TILESIZE,TILESIZE);
    dim3 grid1(updiv(img_width, TILESIZE), updiv(img_height, TILESIZE));
    dim3 grid2(updiv(img_width, TILESIZE*3), updiv(img_height, TILESIZE*3));

    RGB2HSI<<<grid1, block>>>(d_rgb_img, d_hsi_img, img_height, img_width);
    CLAHE<<<grid2, block>>>(d_hsi_img, img_height, img_width);
    HSI2RGB<<<grid1, block>>>(d_hsi_img, d_rgb_img, img_height, img_width);

    cudaMemcpy(ret_img, d_rgb_img, img_height*img_width*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_rgb_img);
    cudaFree(d_hsi_img);
    free(h_img_one);
    free(h_img_two);
    return ret_img;
}
