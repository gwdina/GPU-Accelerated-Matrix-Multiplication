# GPU-Accelerated-Matrix-Multiplication
This project demonstrates matrix multiplication accelerated on a GPU using NVIDIA's CUDA C++ programming model. It is designed to compute the product of two large matrices in parallel, taking advantage of the GPU’s massive threading capabilities to significantly outperform traditional CPU-based matrix multiplication.


Key Features
- Parallel computation of matrix multiplication on the GPU.

- CUDA kernels optimized with 2D thread blocks and grid dimensions.

- Simple interface to demonstrate performance on large matrix sizes (e.g., 1024x1024).

- Easily customizable for different matrix sizes or kernel optimizations.

How It Works:

Matrix Initialization:
Two square matrices A and B are randomly initialized on the host (CPU).

Memory Allocation on Device (GPU):
The matrices are copied from host to device memory using cudaMemcpy.

Kernel Execution:
The matrixMultiply CUDA kernel is launched with a 2D grid of 2D blocks, each computing a subset of the output matrix C.

Thread Mapping:
Each CUDA thread computes one element of matrix C by performing the dot product of a row from A and a column from B.

Result Transfer:
Once computation completes, the result matrix C is copied back from GPU to host memory.

Cleanup:
All allocated memory (both on host and device) is freed to avoid leaks.

Make sure you have the following:
CUDA Toolkit installed

NVIDIA GPU with CUDA support

C++ Compiler (e.g., nvcc) (make sure the correct paths are set up)

How to run in terminal:
nvcc -o matrix_mul gpu_matrix_multiplication.cu
./matrix_mul
