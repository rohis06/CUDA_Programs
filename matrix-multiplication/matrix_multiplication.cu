#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1000  // Size of the matrix (N x N)

__global__ void matrix_mul(float *C, float *A, float *B, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += A[row * n + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Initialize host matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Transfer data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Executing kernel with a single thread
    matrix_mul<<<1, 1>>>(d_C, d_A, d_B, N);

    // Transfer the result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N * N; i++) {
        if (C[i] != N * 2.0f) {
            printf("Error: element C[%d] = %f\n", i, C[i]);
            return -1;
        }
    }

    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Deallocate host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
