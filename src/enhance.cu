#include "enhance.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>

// int updiv(int x, int y) { return (x + y - 1) / y; }

__global__ void RGB2HSI(int * rgb_img, int * hsi_img, int height, int width)
{
    int img_x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x,
      img_y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    int img_idx = __umul24(img_y, width) + img_x;
    if (img_x < width && img_y < height) {
        int R = rgb_img[img_idx] >> 16, G = (rgb_img[img_idx] >> 8) & 0x00FF, B = rgb_img[img_idx] & 0x0000FF;
        float theta = acosf(((R-G+R-B)/2)/sqrtf(powf(R-G, 2)+(R-B)*(G-B)));
        float H = (B <= G)? theta:2*CUDART_PI_F-theta;
        H /= 2*CUDART_PI_F;
        float S = 1 - 3.0*fminf(R,fminf(G, B))/(R+G+B);
        float I = (R+G+B)/(3.0*255);
        hsi_img[img_idx] = (int((H*255))<<16)+(int((S*255))<<8)+int((I*255));
    }
}
__global__ void HSI2RGB(int * hsi_img, int * rgb_img, int height, int width)
{
    int img_x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x,
      img_y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    int img_idx = __umul24(img_y, width) + img_x;
    if (img_x < width && img_y < height) {
        float H = (hsi_img[img_idx] >> 16)/255.0*2*CUDART_PI_F, S = ((hsi_img[img_idx] >> 8) & 0x00FF)/255.0, I = (hsi_img[img_idx] & 0x0000FF)/255.0;
        int R, G, B;
        if(H >= 0 && H < 2*CUDART_PI_F/3) {
            B = I*(1-S)*255;
            R = I*(1+S*cosf(H)/cosf(CUDART_PI_F/3-H))*255;
            G = 3*I*255 - (R+B);
            // printf("1 %d %d %d %f %f %f\n", R, G, B, H, S, I);
        }
        else if(H >= 2*CUDART_PI_F/3 && H < 4*CUDART_PI_F/3) {
            H -= CUDART_PI_F/3*2;
            R = I*(1-S)*255;
            G = I*(1+S*cosf(H)/cosf(CUDART_PI_F/3-H))*255;
            B = 3*I*255 - (R+G);
            // printf("2 %d %d %d %f %f %f\n", R, G, B, H, S, I);
        }
        else if(H >= 4*CUDART_PI_F/3 && H < 2*CUDART_PI_F) {
            H -= CUDART_PI_F/3*4;
            G = I*(1-S)*255;
            B = I*(1+S*cosf(H)/cosf(CUDART_PI_F/3-H))*255;
            R = 3*I*255 - (G+B);
            // printf("3 %d %d %d %f %f %f\n", R, G, B, H, S, I);
        }
        rgb_img[img_idx] = (R << 16) + (G << 8) + B;
    }
}
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

    // if(thread_idx < 256) {
    //     for (int i = 0; i < 9; ++i)
    //     {
    //         int overflow = (frq[i*256+thread_idx] > THRESHOLD)? frq[i*256+thread_idx] - THRESHOLD : 0;
    //         frq[i*256+thread_idx] -= overflow;
    //         atomicAdd(&frq[9*256+i], overflow);
    //     }
    // }
    // __syncthreads();

    // if(thread_idx < 256) {
    //     for (int i = 0; i < 9; ++i)
    //     {
    //         frq[i*256+thread_idx] += frq[9*256+i]/256;
    //     }
    // }

    // __syncthreads();

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
    ret_img = (int*)malloc(img_height*img_width*sizeof(int));
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
    return ret_img;
}

int main(int argc, char **argv) {
  int img_height, img_width;

  FILE *fp;
  fp = fopen(argv[1], "r");
  fscanf(fp, "%d%d", &img_height, &img_width);

  int *src_img = (int *)malloc(sizeof(int) * img_height * img_width);
  for (int i = 0; i < img_height * img_width; ++i) {
    fscanf(fp, "%d", &src_img[i]);
  }
  fclose(fp);

  int *enhancedImg = imgCLAHE(src_img, img_height, img_width);
  for (int i = 0; i < img_height*img_width; ++i)
  {
      printf("%d ", enhancedImg[i] >> 16);
  }
  for (int i = 0; i < img_height*img_width; ++i)
  {
      printf("%d ", (enhancedImg[i] >> 8) & 0x00FF);
  }
  for (int i = 0; i < img_height*img_width; ++i)
  {
      printf("%d ", enhancedImg[i] & 0x0000FF);
  }
  free(enhancedImg);
  return 0;
}
