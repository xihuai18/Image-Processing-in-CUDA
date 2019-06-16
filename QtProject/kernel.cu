#include <cuda.h>

#include <cuda_runtime.h>

#include <stdio.h>


__global__ void addAry( int * ary1, int * ary2 ){
    int indx = threadIdx.x;
    ary1[ indx ] += ary2[ indx ];
}


// Main cuda function
int* getCutMask(int *input_array, int *seed_array, int img_height, int img_width){
    printf("BEGIN CUT");
    return seed_array;
}
