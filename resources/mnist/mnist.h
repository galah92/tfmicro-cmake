#ifndef MNIST_H
#define MNIST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MNIST_N_SAMPLES 10000
#define MNIST_IMG_X 28
#define MNIST_IMG_Y 28
#define MNIST_IMG_SIZE (MNIST_IMG_X * MNIST_IMG_Y)

void mnist_get_model(unsigned char **model_buf, size_t *model_size);
void mnist_get_samples(float **x, uint8_t **y);

#ifdef __cplusplus
}
#endif

#endif  // MNIST_H
