#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "fileloader.hu"

const char *folder = "MNIST-data/";
const char *test_images = "train-images.idx3-ubyte";
const char *test_labels = "train-labels.idx1-ubyte";

void
normalizeArray(float *array, int array_size)
{
  int i;
  float max = -1;
  for (i = 0; i < array_size; i++) {
    if (array[i] > max) max = array[i];
  }
  for (i = 0; i < array_size; i++) {
    array[i] /= max;
  }
}

int
isLittleEndian()
{
  short number = 0x1;
  char *numPtr = (char*)&number;
  return (numPtr[0] == 1);
}

char *
getFilename(int filenum)
{
  char *ret = (char *) malloc(100 * sizeof(char));
  strcat(ret, folder);
  strcat(ret, (filenum == TEST_IMAGES) ? test_images :
              test_labels);
  return ret;
}

void
convertToLittleEndian(int *val)
{
  unsigned char *buffer = (unsigned char*) val;
  char tmp;
  tmp = buffer[0];
  buffer[0] = buffer[3];
  buffer[3] = tmp;
  tmp = buffer[1];
  buffer[1] = buffer[2];
  buffer[2] = tmp;
}

int
labelFile(int filenum)
{
  return filenum == TEST_LABELS;
}

void *
readFile(int filenum)
{
  FILE *fp;
  char *fn;
  fn = getFilename(filenum);
  printf("Reading File: %s\n",fn);
  fp = fopen(fn, "r");
  if (fp == NULL) {
    printf("Error opening file.\n");
    free(fn);
    exit(EXIT_FAILURE);
  }


  char *magic_number;
  int chars_to_read = 4;
  magic_number = (char *) malloc(sizeof(char) * chars_to_read);
    fread(magic_number, sizeof(char), 4, fp);
  printf("Type=%hhu\n",magic_number[2]);
  int num_dimensions = (int) magic_number[3];
  printf("Dimensions=%hhu\n",num_dimensions);


  int *dimensions;
  dimensions = (int *) malloc(sizeof(int) * num_dimensions);
  fread(dimensions, sizeof(int), num_dimensions, fp);
  int i;
  size_t matrix_size = 1;
  for (i = 0; i < num_dimensions; i++) {
    convertToLittleEndian(&dimensions[i]);
    printf("Dim %d has length=%d\n", i, dimensions[i]);
    matrix_size *= dimensions[i];
  }
  printf("Total size is %ld\n", matrix_size);


  unsigned char *dst;
  dst = (unsigned char *) malloc(sizeof(unsigned char) * matrix_size);
  fread(dst, sizeof(unsigned char), matrix_size, fp);
  float *tmp;
  tmp = (float *) malloc(sizeof(float) * matrix_size
                        * (labelFile(filenum) ? 10 : 1));
  for (i = 0; i < matrix_size; i++) {
    if (labelFile(filenum)) {
      tmp[i*10 + (int) dst[i]] = 1;
    }
    else {
      tmp[i] = (float) dst[i];
    }
  }
  if (!labelFile(filenum)) normalizeArray(tmp, matrix_size);

  free(magic_number);
  free(dimensions);
  free(dst);
  fclose(fp);
  free(fn);
  return tmp;
}
