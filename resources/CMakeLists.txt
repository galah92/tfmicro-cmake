add_library(resources
    mnist/mnist.cpp
)

target_include_directories(resources PUBLIC .)
