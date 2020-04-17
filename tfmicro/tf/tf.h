#ifndef TF_H
#define TF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    TF_CODE_SUCCESS,
    TF_CODE_UNINITIALIZED,
    TF_CODE_INVALID_SCHEMA,
    TF_CODE_ALLOC_FAILD,
    TF_CODE_PRED_FAILD,
} TF_CODE;

int tf_init(const void *model_buf, uint8_t *mem, size_t mem_size);
int tf_get_params(void **input, void **output);
int tf_predict();

#ifdef __cplusplus
}
#endif

#endif  // TF_H
