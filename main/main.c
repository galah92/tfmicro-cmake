#include "tf/tf.h"

#include "mnist/mnist.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TF_MEM_SIZE (8 * 1024)

void DebugLog(const char* str)
{
    printf("%s\n", str);
}

bool tf_check(int ts, int line)
{
    bool is_success = ts == TF_CODE_SUCCESS;
    if (!is_success)
    {
        printf("line=%d, ts=%d\n", line, ts);
    }
    return is_success;
}
#define TF_CHECK(ts) do { if (!tf_check(ts, __LINE__)) { return 1; } } while (false)

int main()
{
    float *mnist_x;
    uint8_t *mnist_y;
    mnist_get_samples(&mnist_x, &mnist_y);

    unsigned char *mnist_model_buf;
    size_t mnist_model_size;
    mnist_get_model(&mnist_model_buf, &mnist_model_size);

    static uint8_t tf_mem[TF_MEM_SIZE];
    TF_CHECK(tf_init(mnist_model_buf, tf_mem, TF_MEM_SIZE));        

    float *x_test;
    float *y_test;
    TF_CHECK(tf_get_params((void**)&x_test, (void**)&y_test));

    const size_t n_samples = MNIST_N_SAMPLES;
    size_t n_correct = 0;
    for (size_t i = 0; i < n_samples; i++)
    {
        memcpy(x_test, mnist_x + i * MNIST_IMG_SIZE, MNIST_IMG_SIZE * sizeof(float));
        TF_CHECK(tf_predict());
        size_t pred_i = 0;
        for (size_t j = 0; j < 10; j++)
        {
            pred_i = y_test[j] > y_test[pred_i] ? j : pred_i;
        }
        n_correct += mnist_y[i] == pred_i;
    }
    printf("accuracy=%+6.2f\n", (float)n_correct / n_samples);

    return 0;
}

#ifdef __cplusplus
}
#endif
