#include "tf/tf.h"

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define TF_CHECK(pred, code) do { if (!(pred)) { return code; } } while (false)

static tflite::MicroInterpreter *interpreter = nullptr;

int tf_init(const void *model_buf, uint8_t *mem, size_t mem_size)
{
    static tflite::MicroErrorReporter mer;
    static tflite::ErrorReporter *er = &mer;
    static tflite::ops::micro::AllOpsResolver resolver;

    static const tflite::Model *model = tflite::GetModel(model_buf);
    TF_CHECK(model->version() == TFLITE_SCHEMA_VERSION, TF_CODE_INVALID_SCHEMA);

    static tflite::MicroInterpreter s_interpreter(model, resolver, mem, mem_size, er);
    interpreter = &s_interpreter;

    TfLiteStatus status = interpreter->AllocateTensors();
    TF_CHECK(status == kTfLiteOk, TF_CODE_ALLOC_FAILD);

    return TF_CODE_SUCCESS;
}

int tf_get_params(void **input, void **output)
{
    TF_CHECK(interpreter != nullptr, TF_CODE_UNINITIALIZED);

    TfLiteTensor *input_tensor = interpreter->input(0);
    *input = input_tensor->data.data;

    TfLiteTensor *output_tensor = interpreter->output(0);
    *output = output_tensor->data.data;

    return TF_CODE_SUCCESS;
}

int tf_predict()
{
    TF_CHECK(interpreter != nullptr, TF_CODE_UNINITIALIZED);

    TfLiteStatus status = interpreter->Invoke();
    TF_CHECK(status == kTfLiteOk, TF_CODE_PRED_FAILD);
    return TF_CODE_SUCCESS;
}
