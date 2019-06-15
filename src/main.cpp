#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "build_graph.h"
#include "sharpen.h"
#include "blur.h"

int main(int argc, char **argv) {
  int img_height, img_width;

  FILE *fp;
  fp = fopen(argv[1], "r");
  fscanf(fp, "%d%d", &img_height, &img_width);

  int *src_img = (int *)malloc(sizeof(int) * img_height * img_width);
  int *mask_img = (int *)malloc(sizeof(int) * img_height * img_width);
  for (int i = 0; i < img_height * img_width; ++i) {
    fscanf(fp, "%d", &src_img[i]);
  }
  for (int i = 0; i < img_height * img_width; ++i) {
    fscanf(fp, "%d", &mask_img[i]);
  }
  fclose(fp);

  int *segment = imgBlur(src_img, img_height, img_width);
  for (int j = 0; j < img_width; ++j) {
    for (int i = 0; i < img_height; ++i) {
      printf("%d ", segment[i * img_width + j]);
    }
    printf("\n");
  }

  free(segment);
  free(src_img);
  free(mask_img);

  return 0;
}
