#include <stdio.h>

// Kernel function to add the elements of two vectors
__global__ void vectorAdd(const float* A, const float* B, float* C, int n)
{
    // Get the thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int n = 100000; // Size of the vectors
    const int size = n * sizeof(float);

    // Allocate memory on the host
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize the vectors on the host
    for (int i = 0; i < n; ++i)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate memory on the device
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set the number of threads and blocks
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the first 10 elements of the result
    printf("Result:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // Free memory on the host and device
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
