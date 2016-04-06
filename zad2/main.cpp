#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <algorithm>
#include "reverse.h"


using namespace std;
int main() {
    int n;
    cin >> n;
    vector<int> numbers(n);
    for (int i = 0 ; i < n ; i++) {
        numbers[i] = random();
    }
    int *d_numbers;
    cudaMalloc((void **)&d_numbers, n * sizeof(int));
    cudaMemcpy(d_numbers, numbers.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (n/2 + blockSize - 1) / blockSize;
    reverse<<<blockSize, gridSize>>>(d_numbers, n);

    std::vector<int> output(n);
    cudaMemcpy(d_numbers, output.data(), n * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int>reversed = numbers;
    std::reverse(reversed.begin(), reversed.end());

    if (reversed != output) {
        std::cerr << "output differs\n";
        for (int i = 0 ; i < n ; i++) {
            if (reversed[i] != output[i]) {
                std::cerr << "at location [" << i << "] " << reversed[i]
                    << " != " << output[i] << std::endl;
                break;
            }
        }
    }
    else
        std::cerr<< "OK\n";


    return 0;
}