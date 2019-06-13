#include "onecut_kernel.h"

// __device__ bool fgreater(float x, float y) { return x - y > EPS; }

// __device__ bool fless(float x, float y) { return x - y < -EPS; }

// __device__ bool fequal(float x, float y) { return fabsf(x - y) < EPS; }

// __device__ static float atomicCAS(float* address, float compared, float val) {
//   int* address_as_i = (int*)address;
//   return __int_as_float(
//       ::atomicCAS(address_as_i, __float_as_int(compared), __float_as_int(val)));
// }

__global__ void kernel_pixel_push(unsigned int* res_pixel, unsigned long long* bin_flow,
                                  unsigned int* pixel_flow, unsigned int* pull_pixel,
                                  int* pixel_height, int* bin_height,
                                  int img_size, int col, int row, int tile_size,
                                  int tile_col, int tile_row, int bin_num) {
  // pixel-push->pull_pixel,bin
  int img_x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x,
      img_y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int img_idx = __umul24(img_y, col) + img_x;

  // can use shared memory
  extern __shared__ unsigned int local_res_pixel[];

  // the block size should be the size of the tile
  int thread_idx = threadIdx.x + __umul24(blockDim.x, threadIdx.y);
  int tile_idx = threadIdx.x + 1 + __umul24(tile_col, threadIdx.y + 1);
  if (img_x < col && img_y < row) {
    for (int i = 0; i < RES_UNIT_SIZE; ++i) {
      local_res_pixel[tile_idx * RES_UNIT_SIZE + i] =
          res_pixel[img_idx * RES_UNIT_SIZE + i];
    }

    if (threadIdx.x == 0 && img_x > 0) {
      for (int i = 0; i < RES_UNIT_SIZE; ++i)
        local_res_pixel[(tile_idx - 1) * RES_UNIT_SIZE + i] =
            res_pixel[(img_idx - 1) * RES_UNIT_SIZE + i];
    }
    if (threadIdx.x == tile_col - 2 && img_x < col - 1) {
      for (int i = 0; i < RES_UNIT_SIZE; ++i)
        local_res_pixel[(tile_idx + 1) * RES_UNIT_SIZE + i] =
            res_pixel[(img_idx + 1) * RES_UNIT_SIZE + i];
    }
    if (threadIdx.y == 0 && img_y > 0) {
      for (int i = 0; i < RES_UNIT_SIZE; ++i)
        local_res_pixel[(tile_idx - tile_col) * RES_UNIT_SIZE + i] =
            res_pixel[(img_idx - col) * RES_UNIT_SIZE + i];
    }
    if (threadIdx.y == tile_row - 2 && img_y < row - 1) {
      for (int i = 0; i < RES_UNIT_SIZE; ++i)
        local_res_pixel[(tile_idx + tile_col) * RES_UNIT_SIZE + i] =
            res_pixel[(img_idx + col) * RES_UNIT_SIZE + i];
    }
  }

  __syncthreads();

  unsigned int max_flow_push = pixel_flow[img_idx], min_flow_push = 0, tmp_res;
  int tmp_idx;
  // the thread in the img_block
  if (img_x < col && img_y < row) {
    // to the sink
    tmp_res = local_res_pixel[tile_idx * RES_UNIT_SIZE + 1];
    min_flow_push = max_flow_push;
    if (tmp_res > 0 && max_flow_push > 0 &&
        pixel_height[img_idx] == 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      res_pixel[img_idx * RES_UNIT_SIZE + 1] = tmp_res - min_flow_push;
      pixel_flow[img_idx] -= min_flow_push;
    }
    // to the source
    tmp_res = local_res_pixel[tile_idx * RES_UNIT_SIZE + 0];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    if (tmp_res > 0 && max_flow_push > 0 &&
        pixel_height[img_idx] == bin_height[bin_num] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      res_pixel[img_idx * RES_UNIT_SIZE + 0] = tmp_res - min_flow_push;
      res_pixel[img_idx * RES_UNIT_SIZE + 8] += min_flow_push;
      pixel_flow[img_idx] -= min_flow_push;
      atomicAdd(&bin_flow[bin_num], min_flow_push);
    }
    // bin
    int bin_idx = local_res_pixel[tile_idx * RES_UNIT_SIZE + 6];
    tmp_res = local_res_pixel[tile_idx * RES_UNIT_SIZE + 7];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;

    // printf("%d %d\n", img_idx, tmp_idx);
    // //printf("%d %d\n", img_idx, tmp_idx);
    if (tmp_res > 0 && max_flow_push > 0 &&
        pixel_height[img_idx] == bin_height[bin_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      res_pixel[img_idx * RES_UNIT_SIZE + 7] = tmp_res - min_flow_push;
      res_pixel[img_idx * RES_UNIT_SIZE + 9] += min_flow_push;
      pixel_flow[img_idx] -= min_flow_push;
      atomicAdd(&bin_flow[bin_idx], min_flow_push);
    }

    // up down left right
    tmp_idx = __umul24(img_y - 1, col) + img_x;
    tmp_res = local_res_pixel[tile_idx * RES_UNIT_SIZE + 2];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    if (tmp_idx >= 0 && tmp_idx < img_size &&
        tmp_res > 0 && max_flow_push > 0 &&
        pixel_height[img_idx] == pixel_height[tmp_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      atomicSub(&res_pixel[img_idx * RES_UNIT_SIZE + 2], min_flow_push);
      atomicAdd(&res_pixel[tmp_idx * RES_UNIT_SIZE + 3], min_flow_push);
      atomicSub(&pixel_flow[img_idx], min_flow_push);
      atomicAdd(&pull_pixel[tmp_idx], min_flow_push);
    }

    tmp_idx = (img_y + 1) * col + img_x;
    tmp_res = local_res_pixel[tile_idx * RES_UNIT_SIZE + 3];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    // printf("%d %d %0.2f %0.2f\n", img_idx, tmp_idx, tmp_res, max_flow_push);
    if (tmp_idx >= 0 && tmp_idx < img_size &&
        tmp_res > 0 && max_flow_push > 0 &&
        pixel_height[img_idx] == pixel_height[tmp_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      atomicSub(&res_pixel[img_idx * RES_UNIT_SIZE + 3], min_flow_push);
      atomicAdd(&res_pixel[tmp_idx * RES_UNIT_SIZE + 2], min_flow_push);
      atomicSub(&pixel_flow[img_idx], min_flow_push);
      atomicAdd(&pull_pixel[tmp_idx], min_flow_push);
    }

    tmp_idx = __umul24(img_y, col) + img_x - 1;
    tmp_res = local_res_pixel[tile_idx * RES_UNIT_SIZE + 4];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    // printf("%d %d\n", img_idx, tmp_idx);
    if (tmp_idx >= 0 && tmp_idx < img_size &&
        tmp_res > 0 && max_flow_push > 0 &&
        pixel_height[img_idx] == pixel_height[tmp_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      atomicSub(&res_pixel[img_idx * RES_UNIT_SIZE + 4], min_flow_push);
      atomicAdd(&res_pixel[tmp_idx * RES_UNIT_SIZE + 5], min_flow_push);
      atomicSub(&pixel_flow[img_idx], min_flow_push);
      atomicAdd(&pull_pixel[tmp_idx], min_flow_push);
    }

    tmp_idx = __umul24(img_y, col) + img_x + 1;
    tmp_res = local_res_pixel[tile_idx * RES_UNIT_SIZE + 5];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    // printf("%d %d\n", img_idx, tmp_idx);
    if (tmp_idx >= 0 && tmp_idx < img_size &&
        tmp_res > 0 && max_flow_push > 0 &&
        pixel_height[img_idx] == pixel_height[tmp_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      atomicSub(&res_pixel[img_idx * RES_UNIT_SIZE + 5], min_flow_push);
      atomicAdd(&res_pixel[tmp_idx * RES_UNIT_SIZE + 4], min_flow_push);
      atomicSub(&pixel_flow[img_idx], min_flow_push);
      atomicAdd(&pull_pixel[tmp_idx], min_flow_push);
    }
  }

  // bin-pull->pixel
  // atomicCAS();
  if (img_x < col && img_y < row) {
    // bin
    int bin_idx = res_pixel[img_idx * RES_UNIT_SIZE + 6];
    unsigned int max_flow_pull = res_pixel[img_idx * RES_UNIT_SIZE + 9];
    unsigned long long bin_res_flow = 0;
    unsigned int flow_pull = 0;
    unsigned long long new_bin_flow = 0;
    do {
      bin_res_flow = bin_flow[bin_idx];
      if (max_flow_pull > 0 &&
          bin_res_flow > 0 &&
          bin_height[bin_idx] == pixel_height[img_idx] + 1) {
        flow_pull =
            max_flow_pull < bin_res_flow ? max_flow_pull : bin_res_flow;
        new_bin_flow = bin_res_flow - flow_pull;
      } else {
        flow_pull = 0;
        break;
      }
    } while (bin_res_flow != atomicCAS(&bin_flow[bin_idx], bin_res_flow,
                                             new_bin_flow));
    pixel_flow[img_idx] += flow_pull;
    res_pixel[img_idx * RES_UNIT_SIZE + 9] -= flow_pull;
    res_pixel[img_idx * RES_UNIT_SIZE + 7] += flow_pull;

    // source
    max_flow_pull = res_pixel[img_idx * RES_UNIT_SIZE + 8];
    do {
      bin_res_flow = bin_flow[bin_num];
      if (max_flow_pull > 0 &&
          bin_res_flow > 0 &&
          bin_height[bin_num] == pixel_height[img_idx] + 1) {
        flow_pull =
            max_flow_pull < bin_res_flow ? max_flow_pull : bin_res_flow;
        new_bin_flow = bin_res_flow - flow_pull;
      } else {
        flow_pull = 0;
        break;
      }
    } while (bin_res_flow != atomicCAS(&bin_flow[bin_num], bin_res_flow,
                                             new_bin_flow));
    pixel_flow[img_idx] += flow_pull;
    res_pixel[img_idx * RES_UNIT_SIZE + 8] -= flow_pull;
    res_pixel[img_idx * RES_UNIT_SIZE + 0] += flow_pull;
  }
}

__global__ void kernel_pixel_pull(unsigned int* res_pixel, unsigned int* pull_pixel,
                                  unsigned int* pixel_flow, int img_size, int col,
                                  int row) {
  // pixel<-pull-pull_pixel
  int img_x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x,
      img_y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int img_idx = __umul24(img_y, col) + img_x;
  if (img_x < col && img_y < row) {
    pixel_flow[img_idx] += pull_pixel[img_idx];
    pull_pixel[img_idx] = 0;
  }
}

__global__ void kernel_pixel_relabel(unsigned int* res,
                                     unsigned int* pixel_flow,
                                     int* pixel_height, int* bin_height,
                                     int img_size, int col, int row,
                                     int tile_size, int tile_col, int tile_row,
                                     int bin_num, bool* finished) {
  /*
  Each threads handles one pixel. For each pixel with positive flow, find the
  minimum height of potential pixels/bins (with positive residual), set its
  height to min_height + 1.

  Size of shared memory: 4 * (tile_size + bin_num + 1)

  NOTES FOR OPTIMIZATION:
  1. swap the row & col of res may be better? (spacial locality)
  2. consider texture
  */

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = x * col + y;
  int block_pid = threadIdx.y * blockDim.x + threadIdx.x;
  int tile_pid = (threadIdx.y + 1) * tile_col + threadIdx.x + 1;

  // load pixel_height to shared memory
  extern __shared__ int height[];
  height[tile_pid] = (x < row && y < col) ? pixel_height[pid] : INF;
  if (threadIdx.y == 0) {
    height[tile_pid - tile_col] =
        (x > 0 && y < col) ? pixel_height[pid - col] : INF;
  }
  if (threadIdx.y == blockDim.y - 1) {
    height[tile_pid + tile_col] =
        (x + 1 < row && y < col) ? pixel_height[pid + col] : INF;
  }
  if (threadIdx.x == 0) {
    height[tile_pid - 1] = (y > 0 && x < row) ? pixel_height[pid - 1] : INF;
  }
  if (threadIdx.x == blockDim.x - 1) {
    height[tile_pid + 1] =
        (y + 1 < col && x < row) ? pixel_height[pid + 1] : INF;
  }
  // if (block_pid <= bin_num) {
  //   height[tile_size + block_pid] = bin_height[block_pid];
  // }
  __syncthreads();

  // relabel
  if (x < row && y < col && pixel_flow[pid] > 0) {
    int min_height = INF;
    unsigned int* res_pixel = res + pid * RES_UNIT_SIZE;
    // pixel -> S
    if (res_pixel[0] > 0) {
      // min_height = min(min_height, height[tile_size + bin_num]);
      min_height = bin_height[bin_num];
    }
    // pixel -> T
    if (res_pixel[1] > 0) {
      min_height = 0;
    }
    // pixel -> pixel
    if (res_pixel[2] > 0) {  // up
      min_height = min(min_height, height[tile_pid - tile_col]);
    }
    if (res_pixel[3] > 0) {  // down
      min_height = min(min_height, height[tile_pid + tile_col]);
    }
    if (res_pixel[4] > 0) {  // left
      min_height = min(min_height, height[tile_pid - 1]);
    }
    if (res_pixel[5] > 0) {  // right
      min_height = min(min_height, height[tile_pid + 1]);
    }
    // pixel -> bin
    if (res_pixel[7] > 0) {
      // min_height = min(min_height, height[tile_size + res_pixel[6]]);
      min_height = min(min_height, bin_height[res_pixel[6]]);
    }
    if (min_height < INF) {
      pixel_height[pid] = min_height + 1;
      *finished = false;
    }
  }
}

__global__ void kernel_bin_relabel(unsigned int* res, unsigned int* pixel_flow,
                                   unsigned long long* bin_flow,
                                   int* pixel_height, int* bin_height,
                                   int img_size, int col, int row,
                                   int tile_size, int tile_col, int tile_row,
                                   int bin_num, bool* finished) {
  /*
  Each thread handles one pixel. For each pixel, relabel the bin connected
  to it, if the bin has positive flow and the pixel is pushable.

  NOTES FOR OPTIMIZATION:
  1. use reduction rather than atomic ops
  */

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = x * col + y;
  if (x < row && y < col) {
    int new_height = pixel_height[pid] + 1 - INF;
    int read_height, returned_height;
    int* target_height;
    // bin
    int bid = res[pid * RES_UNIT_SIZE + 6];
    if (bin_flow[bid] > 0) {
      if (res[pid * RES_UNIT_SIZE + 9] > 0) {
        target_height = bin_height + bid;
        do {
          read_height = *target_height;
          returned_height =
              (read_height > new_height)
                  ? atomicCAS(target_height, read_height, new_height)
                  : read_height;
        } while (read_height != returned_height);
      }
    }
  }
}

__global__ void kernel_bin_relabel_rectify(int* bin_height, int bin_num,
                                           bool* finished) {
  /*
  Each thread handles one bin (or S). Let height += INF if it is negative
  (which means it has been relabeled).
  */

  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (bid < bin_num && bin_height[bid] < 0) {
    bin_height[bid] += INF;
    *finished = false;
  }
}

__global__ void kernel_bfs_init(unsigned int* res, int* bfs_pixel_height,
                                int* bfs_bin_height, int img_size, int col,
                                int row, int bin_num) {
  /*
  Each thread handles one pixel. For each pixel, if it is reachable from
  S in the res graph, set height=1; if it can reach T, set height=-1;
  otherwise, set height=INF.
  Set height=INF for all bins.

  NOTES FOR OPTIMIZATION:
  */

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = x * col + y;

  if (x < row && y < col) {
    bfs_pixel_height[pid] = (res[pid * RES_UNIT_SIZE + 8] > 0) ? 1 : INF;
  }
  if (pid < bin_num) {
    bfs_bin_height[pid] = INF;
  }
}

__global__ void kernel_pixel_bfs(unsigned int* res, int* bfs_pixel_height,
                                 int* bfs_bin_height, int img_size, int col,
                                 int row, int tile_size, int tile_col,
                                 int tile_row, int bin_num, int cur_height,
                                 bool* finished) {
  /*
  Each thread handles one pixel. For each pixel, if its height is cur_height,
  set the height of reachable & unvisited points/bin to be cur_height + 1.
  If any height assignment is performed, set finished = false.

  Size of shared memory: 4 * (tile_size + bin_num + 1)

  NOTES FOR OPTIMIZATION:
  */

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = x * col + y;
  int block_pid = threadIdx.y * blockDim.x + threadIdx.x;
  int tile_pid = (threadIdx.y + 1) * tile_col + threadIdx.x + 1;

  // load bfs_pixel_height to shared memory
  extern __shared__ int height[];
  height[tile_pid] = (x < row && y < col) ? bfs_pixel_height[pid] : -INF;
  if (threadIdx.y == 0) {
    height[tile_pid - tile_col] =
        (x > 0 && y < col) ? bfs_pixel_height[pid - col] : -INF;
  }
  if (threadIdx.y == blockDim.y - 1) {
    height[tile_pid + tile_col] =
        (x + 1 < row && y < col) ? bfs_pixel_height[pid + col] : -INF;
  }
  if (threadIdx.x == 0) {
    height[tile_pid - 1] =
        (y > 0 && x < row) ? bfs_pixel_height[pid - 1] : -INF;
  }
  if (threadIdx.x == blockDim.x - 1) {
    height[tile_pid + 1] =
        (y + 1 < col && x < row) ? bfs_pixel_height[pid + 1] : -INF;
  }
  // if (block_pid <= bin_num) {
  //   height[tile_size + block_pid] = bfs_bin_height[block_pid];
  // }
  __syncthreads();

  // expand
  if (x < row && y < col && height[tile_pid] == cur_height) {
    unsigned int* res_pixel = res + pid * RES_UNIT_SIZE;
    bool modified = false;
    if (res_pixel[2] > 0 && height[tile_pid - tile_col] > cur_height) {  // up
      bfs_pixel_height[pid - col] = cur_height + 1;
      modified = true;
    }
    if (res_pixel[3] > 0 && height[tile_pid + tile_col] > cur_height) {  // down
      bfs_pixel_height[pid + col] = cur_height + 1;
      modified = true;
    }
    if (res_pixel[4] > 0 && height[tile_pid - 1] > cur_height) {  // left
      bfs_pixel_height[pid - 1] = cur_height + 1;
      modified = true;
    }
    if (res_pixel[5] > 0 && height[tile_pid + 1] > cur_height) {  // right
      bfs_pixel_height[pid + 1] = cur_height + 1;
      modified = true;
    }
    int bid = res_pixel[6];
    // if (res_pixel[7] > 0 && height[tile_size + bid] > cur_height) {  // bin
    if (res_pixel[7] > 0 && bfs_bin_height[bid] > cur_height) {  // bin
      bfs_bin_height[bid] = cur_height + 1;
      modified = true;
    }
    if (modified) {
      *finished = false;
    }
  }
}

__global__ void kernel_bin_bfs(unsigned int* res, int* bfs_pixel_height,
                               int* bfs_bin_height, int img_size, int col,
                               int row, int bin_num, int cur_height,
                               bool* finished) {
  /*
  Each thread handles one pixel. For each pixel, if its height > cur_height
  and the height of connected bin is cur_height and the pixel is reachable
  from the bin, set pixel height to be cur_height + 1.
  Set finished = false if assignment happens.

  Size of shared memory: 4 * bin_num

  NOTES FOR OPTIMIZATION:
  */

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = x * col + y;
  int block_pid = threadIdx.y * blockDim.x + threadIdx.x;

  // load bfs_bin_height to shared memory
  extern __shared__ int height[];
  // if (block_pid < bin_num) {
  //   height[block_pid] = bfs_bin_height[block_pid];
  // }
  __syncthreads();

  if (x < row && y < col && bfs_pixel_height[pid] > cur_height) {
    int bid = res[pid * RES_UNIT_SIZE + 6];
    // if (height[bid] == cur_height && res[pid * RES_UNIT_SIZE + 9] > 0) {
    if (bfs_bin_height[bid] == cur_height && res[pid * RES_UNIT_SIZE + 9] > 0) {
      bfs_pixel_height[pid] = cur_height + 1;
      *finished = false;
    }
  }
}

__global__ void kernel_segment(int* bfs_pixel_height, int img_size, int col,
                               int row) {
  /*
  Each thread handles one pixel. For each pixel, set bfs_pixel_height to 0
  if it is INF, otherwise 255.
  */

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = x * col + y;

  if (x < row && y < col) {
    int* height = bfs_pixel_height + pid;
    *height = (*height == INF) ? 0 : 255;
  }
}
