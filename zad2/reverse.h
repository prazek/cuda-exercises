//
// Created by prazek on 03.04.16.
//

#ifndef ZAD2_REVERSE_H
#define ZAD2_REVERSE_H

#include <device_launch_parameters.h>

__global__ void reverse(int *d_input, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
        return;

    int tmp = d_input[index];
    d_input[index] = d_input[n - index - 1];
    d_input[n - index - 1] = tmp;
}



#endif //ZAD2_REVERSE_H
