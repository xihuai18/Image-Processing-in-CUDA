#include "onecut_kernel.h"

__global__ void kernel_pixel_push(unsigned int* res_pixel,
                                  unsigned long long* bin_flow,
                                  unsigned int* pixel_flow,
                                  unsigned int* pull_pixel, int* pixel_height,
                                  int* bin_height, int img_size, int col,
                                  int row, int tile_size, int tile_col,
                                  int tile_row, int bin_num) {
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
      local_res_pixel[tile_idx + tile_size * i] =
          res_pixel[img_idx + img_size * i];
    }

    if (threadIdx.y == 0 && img_y > 0) {
      for (int i = 0; i < RES_UNIT_SIZE; ++i)
        local_res_pixel[(tile_idx - tile_col) + tile_size * i] =
            res_pixel[(img_idx - col) + img_size * i];
    }
    if (threadIdx.y == tile_row - 2 && img_y < row - 1) {
      for (int i = 0; i < RES_UNIT_SIZE; ++i)
        local_res_pixel[(tile_idx + tile_col) + tile_size * i] =
            res_pixel[(img_idx + col) + img_size * i];
    }
    if (threadIdx.x == 0 && img_x > 0) {
      for (int i = 0; i < RES_UNIT_SIZE; ++i)
        local_res_pixel[(tile_idx - 1) + tile_size * i] =
            res_pixel[(img_idx - 1) + img_size * i];
    }
    if (threadIdx.x == tile_col - 2 && img_x < col - 1) {
      for (int i = 0; i < RES_UNIT_SIZE; ++i)
        local_res_pixel[(tile_idx + 1) + tile_size * i] =
            res_pixel[(img_idx + 1) + img_size * i];
    }
  }

  __syncthreads();

  unsigned int max_flow_push = pixel_flow[img_idx], min_flow_push = 0, tmp_res;
  int tmp_idx;
  // the thread in the img_block
  if (img_x < col && img_y < row) {
    // to the sink
    tmp_res = local_res_pixel[tile_idx + tile_size * 1];
    min_flow_push = max_flow_push;
    if (tmp_res > 0 && max_flow_push > 0 && pixel_height[img_idx] == 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      res_pixel[img_idx + img_size * 1] = tmp_res - min_flow_push;
      pixel_flow[img_idx] -= min_flow_push;
    }
    // to the source
    tmp_res = local_res_pixel[tile_idx + tile_size * 0];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    if (tmp_res > 0 && max_flow_push > 0 &&
        pixel_height[img_idx] == bin_height[bin_num] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      res_pixel[img_idx + img_size * 0] = tmp_res - min_flow_push;
      res_pixel[img_idx + img_size * 8] += min_flow_push;
      pixel_flow[img_idx] -= min_flow_push;
      atomicAdd(&bin_flow[bin_num], min_flow_push);
    }
    // bin
    int bin_idx = local_res_pixel[tile_idx + tile_size * 6];
    tmp_res = local_res_pixel[tile_idx + tile_size * 7];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;

    // printf("%d %d\n", img_idx, tmp_idx);
    // //printf("%d %d\n", img_idx, tmp_idx);
    if (tmp_res > 0 && max_flow_push > 0 &&
        pixel_height[img_idx] == bin_height[bin_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      res_pixel[img_idx + img_size * 7] = tmp_res - min_flow_push;
      res_pixel[img_idx + img_size * 9] += min_flow_push;
      pixel_flow[img_idx] -= min_flow_push;
      atomicAdd(&bin_flow[bin_idx], min_flow_push);
    }

    // up down left right
    tmp_idx = __umul24(img_y - 1, col) + img_x;
    tmp_res = local_res_pixel[tile_idx + tile_size * 2];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    if (tmp_idx >= 0 && tmp_idx < img_size && tmp_res > 0 &&
        max_flow_push > 0 &&
        pixel_height[img_idx] == pixel_height[tmp_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      atomicSub(&res_pixel[img_idx + img_size * 2], min_flow_push);
      atomicAdd(&res_pixel[tmp_idx + img_size * 3], min_flow_push);
      atomicSub(&pixel_flow[img_idx], min_flow_push);
      atomicAdd(&pull_pixel[tmp_idx], min_flow_push);
    }

    tmp_idx = (img_y + 1) * col + img_x;
    tmp_res = local_res_pixel[tile_idx + tile_size * 3];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    // printf("%d %d %0.2f %0.2f\n", img_idx, tmp_idx, tmp_res, max_flow_push);
    if (tmp_idx >= 0 && tmp_idx < img_size && tmp_res > 0 &&
        max_flow_push > 0 &&
        pixel_height[img_idx] == pixel_height[tmp_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      atomicSub(&res_pixel[img_idx + img_size * 3], min_flow_push);
      atomicAdd(&res_pixel[tmp_idx + img_size * 2], min_flow_push);
      atomicSub(&pixel_flow[img_idx], min_flow_push);
      atomicAdd(&pull_pixel[tmp_idx], min_flow_push);
    }

    tmp_idx = __umul24(img_y, col) + img_x - 1;
    tmp_res = local_res_pixel[tile_idx + tile_size * 4];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    // printf("%d %d\n", img_idx, tmp_idx);
    if (tmp_idx >= 0 && tmp_idx < img_size && tmp_res > 0 &&
        max_flow_push > 0 &&
        pixel_height[img_idx] == pixel_height[tmp_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      atomicSub(&res_pixel[img_idx + img_size * 4], min_flow_push);
      atomicAdd(&res_pixel[tmp_idx + img_size * 5], min_flow_push);
      atomicSub(&pixel_flow[img_idx], min_flow_push);
      atomicAdd(&pull_pixel[tmp_idx], min_flow_push);
    }

    tmp_idx = __umul24(img_y, col) + img_x + 1;
    tmp_res = local_res_pixel[tile_idx + tile_size * 5];
    max_flow_push = pixel_flow[img_idx];
    min_flow_push = max_flow_push;
    // printf("%d %d\n", img_idx, tmp_idx);
    if (tmp_idx >= 0 && tmp_idx < img_size && tmp_res > 0 &&
        max_flow_push > 0 &&
        pixel_height[img_idx] == pixel_height[tmp_idx] + 1) {
      (tmp_res < max_flow_push) ? min_flow_push = tmp_res : 0;
      atomicSub(&res_pixel[img_idx + img_size * 5], min_flow_push);
      atomicAdd(&res_pixel[tmp_idx + img_size * 4], min_flow_push);
      atomicSub(&pixel_flow[img_idx], min_flow_push);
      atomicAdd(&pull_pixel[tmp_idx], min_flow_push);
    }
  }

  // bin-pull->pixel
  // atomicCAS();
  if (img_x < col && img_y < row) {
    // bin
    int bin_idx = res_pixel[img_idx + img_size * 6];
    unsigned int max_flow_pull = res_pixel[img_idx + img_size * 9];
    unsigned long long bin_res_flow = 0;
    unsigned int flow_pull = 0;
    unsigned long long new_bin_flow = 0;
    do {
      bin_res_flow = bin_flow[bin_idx];
      if (max_flow_pull > 0 && bin_res_flow > 0 &&
          bin_height[bin_idx] == pixel_height[img_idx] + 1) {
        flow_pull = max_flow_pull < bin_res_flow ? max_flow_pull : bin_res_flow;
        new_bin_flow = bin_res_flow - flow_pull;
      } else {
        flow_pull = 0;
        break;
      }
    } while (bin_res_flow !=
             atomicCAS(&bin_flow[bin_idx], bin_res_flow, new_bin_flow));
    pixel_flow[img_idx] += flow_pull;
    res_pixel[img_idx + img_size * 9] -= flow_pull;
    res_pixel[img_idx + img_size * 7] += flow_pull;

    // source
    max_flow_pull = res_pixel[img_idx + img_size * 8];
    do {
      bin_res_flow = bin_flow[bin_num];
      if (max_flow_pull > 0 && bin_res_flow > 0 &&
          bin_height[bin_num] == pixel_height[img_idx] + 1) {
        flow_pull = max_flow_pull < bin_res_flow ? max_flow_pull : bin_res_flow;
        new_bin_flow = bin_res_flow - flow_pull;
      } else {
        flow_pull = 0;
        break;
      }
    } while (bin_res_flow !=
             atomicCAS(&bin_flow[bin_num], bin_res_flow, new_bin_flow));
    pixel_flow[img_idx] += flow_pull;
    res_pixel[img_idx + img_size * 8] -= flow_pull;
    res_pixel[img_idx + img_size * 0] += flow_pull;
  }
}

__global__ void kernel_pixel_pull(unsigned int* res_pixel,
                                  unsigned int* pull_pixel,
                                  unsigned int* pixel_flow, int img_size,
                                  int col, int row) {
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
                                     int* height_count, int* gap, int img_size,
                                     int col, int row, int tile_size,
                                     int tile_col, int tile_row, int bin_num,
                                     bool* finished) {
  /*
  Each threads handles one pixel. For each pixel with positive flow, find the
  minimum height of potential pixels/bins (with positive residual), set its
  height to min_height + 1.

  Size of shared memory: 4 * tile_size

  NOTES FOR OPTIMIZATION:
  1. swap the row & col of res may be better? (spacial locality)
  2. consider texture
  */

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = x * col + y;
  // int block_pid = threadIdx.y * blockDim.x + threadIdx.x;
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
  __syncthreads();

  // relabel
  if (x < row && y < col && pixel_flow[pid] > 0) {
    int min_height = INF;
    // pixel -> S
    if (res[0 * img_size + pid] > 0) {
      min_height = bin_height[bin_num];
    }
    // pixel -> T
    if (res[1 * img_size + pid] > 0) {
      min_height = 0;
    }
    // pixel -> pixel
    if (res[2 * img_size + pid] > 0) {
      min_height = min(min_height, height[tile_pid - tile_col]);
    }
    if (res[3 * img_size + pid] > 0) {
      min_height = min(min_height, height[tile_pid + tile_col]);
    }
    if (res[4 * img_size + pid] > 0) {
      min_height = min(min_height, height[tile_pid - 1]);
    }
    if (res[5 * img_size + pid] > 0) {
      min_height = min(min_height, height[tile_pid + 1]);
    }
    // pixel -> bin
    if (res[7 * img_size + pid] > 0) {
      min_height = min(min_height, bin_height[res[6 * img_size + pid]]);
    }
    int cur_height = height[tile_pid];
    if (min_height < INF) {
      if (min_height >= cur_height) {
        pixel_height[pid] = min_height + 1;
        atomicSub(height_count + cur_height, 1);
        atomicAdd(height_count + min_height + 1, 1);
      }
      *finished = false;
    }
  }
}

__global__ void kernel_bin_relabel(unsigned int* res, unsigned int* pixel_flow,
                                   unsigned long long* bin_flow,
                                   int* pixel_height, int* bin_height,
                                   int* new_bin_height, int img_size, int col,
                                   int row, int tile_size, int tile_col,
                                   int tile_row, int bin_num) {
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
    int bid = res[6 * img_size + pid];
    if (bin_flow[bid] > 0 && res[9 * img_size + pid] > 0) {
      atomicMin(new_bin_height + bid, new_height);
    }
  }
}

__global__ void kernel_bin_relabel_update(int* bin_height, int* new_bin_height,
                                          int* height_count, int bin_num,
                                          int max_height, bool* finished) {
  /*
  Each thread handles one bin. For each bin, update its height if it has
  been changed.
  */

  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (bid < bin_num) {
    int cur_height = bin_height[bid];
    int new_height = new_bin_height[bid];
    if (new_height < 0) {
      new_height += INF;
      if (new_height > cur_height) {
        atomicSub(height_count + cur_height, 1);
        atomicAdd(height_count + new_height, 1);
        bin_height[bid] = new_height;
      }
      new_bin_height[bid] = INF;
      *finished = false;
    }
  }
}

__global__ void kernel_check_gap(int* height_count, int* gap, int num) {
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h < num - 1) {
    if (height_count[h] == 0 && height_count[h + 1] > 0) {
      atomicMin(gap, h);
    }
  }
}

__global__ void kernel_gap_relabel(int* pixel_height, int* bin_height,
                                   int* height_count, int img_size, int bin_num,
                                   int gap) {
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  int new_height = img_size + bin_num + 3;
  if (h < img_size) {
    int cur_height = pixel_height[h];
    if (cur_height > gap && cur_height < new_height) {
      atomicSub(height_count + cur_height, 1);
      atomicAdd(height_count + new_height, 1);
      pixel_height[h] = new_height;
    }
  } else if (h < img_size + bin_num) {
    int cur_height = bin_height[h - img_size];
    if (cur_height > gap && cur_height < new_height) {
      atomicSub(height_count + cur_height, 1);
      atomicAdd(height_count + new_height, 1);
      bin_height[h - img_size] = new_height;
    }
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
    bfs_pixel_height[pid] = (res[8 * img_size + pid] > 0) ? 1 : INF;
  }
  if (pid < img_size) {
    for (int offset = 0; offset < bin_num; offset += img_size) {
      if (offset + pid < bin_num) {
        bfs_bin_height[offset + pid] = INF;
      }
    }
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

  Size of shared memory: 4 * tile_size

  NOTES FOR OPTIMIZATION:
  */

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = x * col + y;
  // int block_pid = threadIdx.y * blockDim.x + threadIdx.x;
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
  __syncthreads();

  // expand
  if (x < row && y < col && height[tile_pid] == cur_height) {
    bool modified = false;
    if (res[2 * img_size + pid] > 0 &&
        height[tile_pid - tile_col] > cur_height) {  // up
      bfs_pixel_height[pid - col] = cur_height + 1;
      modified = true;
    }
    if (res[3 * img_size + pid] > 0 &&
        height[tile_pid + tile_col] > cur_height) {  // down
      bfs_pixel_height[pid + col] = cur_height + 1;
      modified = true;
    }
    if (res[4 * img_size + pid] > 0 &&
        height[tile_pid - 1] > cur_height) {  // left
      bfs_pixel_height[pid - 1] = cur_height + 1;
      modified = true;
    }
    if (res[5 * img_size + pid] > 0 &&
        height[tile_pid + 1] > cur_height) {  // right
      bfs_pixel_height[pid + 1] = cur_height + 1;
      modified = true;
    }
    int bid = res[6 * img_size + pid];
    if (res[7 * img_size + pid] > 0 &&
        bfs_bin_height[bid] > cur_height) {  // bin
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

  NOTES FOR OPTIMIZATION:
  */

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = x * col + y;

  if (x < row && y < col && bfs_pixel_height[pid] > cur_height) {
    int bid = res[6 * img_size + pid];
    if (bfs_bin_height[bid] == cur_height && res[9 * img_size + pid] > 0) {
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
