#include "mnist/mnist.h"

#include <vector>
#include <iostream>
#include <fstream>

void mnist_get_model(unsigned char **model_buf, size_t *model_size)
{
    static std::vector<unsigned char> model;
    if (model.empty())
    {
        std::ifstream is("resources/mnist/q_mnist_model.tflite", std::ios::binary);
        is.seekg(0, is.end);
        model.resize(is.tellg());
        is.seekg(0, is.beg);
        is.read((char*)model.data(), model.size());
    }
    *model_buf = model.data();
}

void mnist_get_samples(float **x, uint8_t **y)
{
    static std::vector<float> x_samples;
    if (x_samples.empty())
    {
        std::ifstream is("resources/mnist/mnist_x_test.bin", std::ios::binary);
        is.seekg(0, is.end);
        x_samples.resize(is.tellg());
        is.seekg(0, is.beg);
        is.read((char*)x_samples.data(), x_samples.size() * sizeof(float));
    }
    *x = x_samples.data();

    static std::vector<uint8_t> y_samples;
    if (y_samples.empty())
    {
        std::ifstream is("resources/mnist/mnist_y_test.bin", std::ios::binary);
        is.seekg(0, is.end);
        y_samples.resize(is.tellg());
        is.seekg(0, is.beg);
        is.read((char*)y_samples.data(), y_samples.size() * sizeof(uint8_t));
    }
    *y = y_samples.data();
}
