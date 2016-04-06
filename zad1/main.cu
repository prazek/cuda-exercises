#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrix_multiplication(const int *d_indices,
                                      const int *d_matrix,
                                      const int *d_vector,
                                      int *d_output,
                                      int n,
                                      int t) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
        return;

    int result = 0;
    for (int i = 0 ; i < t; i++) {
        const int multiplierIndex = index + d_indices[i];
        if (multiplierIndex < 0 || multiplierIndex > n)
            continue;
        const int elemIndex = n * i + index;
        result += d_matrix[elemIndex] * d_vector[multiplierIndex];
    }
    d_output[index] = result;
}

#define cudaCheckErrors(EXPR) assert(EXPR == cudaSuccess)

/// Function for fast integer fetching
int fetch_int() {
    int result = 0;

    char c = 0;
    // skip all other chars
    while (c < '0' or c > '9') {
        c = getchar_unlocked();
    }

    while ('0' <= c and c <= '9') {
        result *= 10;
        result += c - '0';
        c = getchar_unlocked();
    }

    return result;
}


int main() {
    int n = fetch_int();
    int t = fetch_int();
    int *h_fullInput;

    // We need memory for n indices, n * t matrix elements and n elements of vector.
    const int full = n * (t + 2);
    cudaCheckErrors(cudaMallocHost((void**)&h_fullInput, sizeof(int) * full));
    int *h_indices = h_fullInput;
    int *h_matrix = h_fullInput + n;
    int *h_vector = h_matrix + n * t;

    for (int i = 0; i < t; i++) {
        h_indices[i] = fetch_int();
        for (int j = 0 ; j < n ; j++) {
            int index = t * i + j;
            h_matrix[index] = fetch_int();
        }
    }

    for (int i = 0 ; i < n ; i++) {
        h_vector[i] = fetch_int();
    }

    int *d_fullInput;
    cudaCheckErrors(cudaMalloc((void**)&d_fullInput, sizeof(int) * full));
    cudaMemcpy(d_fullInput, h_fullInput, sizeof(int) * full, cudaMemcpyHostToDevice);
    const int *d_indices = d_fullInput;
    const int *d_matrix = d_fullInput + n;
    const int *d_vector = d_matrix + n * t;

    int * d_output;
    cudaCheckErrors(cudaMalloc((void**)&d_output, sizeof(int) * n));

    const int blockSize = 512;
    const int gridSize = (n + blockSize - 1) / blockSize;

    matrix_multiplication<<<gridSize, blockSize>>>(d_indices, d_matrix, d_vector, d_output, n, t);

    // write output into indices to save malloc call
    int * h_output = h_indices;
    cudaMemcpy(h_output, d_output, sizeof(int) * n, cudaMemcpyDeviceToHost);
    for (int i = 0 ; i < n ; i++) {
        printf("%d\n", d_output[i]);
    }

    cudaFree(d_fullInput);
    cudaFree(h_fullInput);
    cudaFree(d_output);
}