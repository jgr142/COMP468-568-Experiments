#pragma once

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32

    inline dim3
    make_grid(int m, int n) {
  return dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
}

__global__ void gemm_naive_kernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int N, int K) {
  /* TODO(student): compute row/col indices, accumulate dot product, write to C
   */
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float sum = 0;
    for (int i{0}; i < K; i++) {
      sum += A[K * row + i] * B[N * i + col];
    }
    C[row * N + col] = sum;
  }
}

__global__ void gemm_tiled_kernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int N, int K) {
  /* TODO(student): use shared memory tiles of size BLOCK_SIZE x BLOCK_SIZE */
  __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];

  // TODO: Loop over K tiles
  // TODO: Each thread loads one element of Tile A and Tile B
  // TODO: Sync the threads so that they do not begin computing until all of the
  // elements of the tiles have been filled
  // TODO: Multiply and sum for a tile of C
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;

  // Loop over all tiles in the K dimension
  for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
    // Load tiles into shared memory
    int tile_col = tile * BLOCK_SIZE + threadIdx.x;
    int tile_row = tile * BLOCK_SIZE + threadIdx.y;

    if (row < M && tile_col < K) {
      tile_A[threadIdx.y][threadIdx.x] = A[row * K + tile_col];
    } else {
      tile_A[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (tile_row < K && col < N) {
      tile_B[threadIdx.y][threadIdx.x] = B[tile_row * N + col];
    } else {
      tile_B[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

// Compute partial sum for this tile
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
      sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
    }

    __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

inline void launch_naive_gemm(const float *d_a, const float *d_b, float *d_c,
                              int M, int N, int K, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid = make_grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  gemm_naive_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
}

inline void launch_tiled_gemm(const float *d_a, const float *d_b, float *d_c,
                              int M, int N, int K, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid = make_grid(((M + BLOCK_SIZE - 1) / BLOCK_SIZE),
                        (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  /* TODO(student): launch gemm_tiled_kernel and check for errors. */
  gemm_tiled_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
}
