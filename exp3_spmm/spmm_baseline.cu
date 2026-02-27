#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

extern void generate_random_csr(int M, int K, double density,
                                std::vector<int> &row_ptr,
                                std::vector<int> &col_idx,
                                std::vector<float> &vals, unsigned seed);

extern void spmm_cpu(int M, int K, int N, const std::vector<int> &row_ptr,
                     const std::vector<int> &col_idx,
                     const std::vector<float> &vals,
                     const std::vector<float> &B, std::vector<float> &C);

extern float max_abs_err(const std::vector<float> &A,
                         const std::vector<float> &B);

using float_t = float;

/*
===============================================================
 BASELINE KERNEL — one thread processes ONE ROW of A
 STUDENT TODO:
   - Fill missing loops
   - Compute C[row, j] += value * B[k, j]
===============================================================
*/
__global__ void spmm_csr_row_kernel(int M, int N,
                                    const int *__restrict__ d_row_ptr,
                                    const int *__restrict__ d_col_idx,
                                    const float_t *__restrict__ d_vals,
                                    const float_t *__restrict__ d_B,
                                    float_t *__restrict__ d_C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  // TODO (student): Initialize output row C[row, :]
  float_t *output_row = d_C + row * N;
  for (int j = 0; j < N; j++) {
    output_row[j] = 0.0f;
  }

  // Find nonzero range
  int start, end;
  start = d_row_ptr[row];
  end = d_row_ptr[row + 1];

  // Loop over nonzeros in this row
  for (int idx = start; idx < end; idx++) {
    int k = d_col_idx[idx];

    float v = d_vals[idx];

    for (int n = 0; n < N; n++) {
      float v2 = d_B[k * N + n];
      output_row[n] += v2 * v;
    }
  }
}

/*
===============================================================
 MAIN PROGRAM
===============================================================
*/
int main(int argc, char **argv) {
  int M = 512, K = 512, N = 64;
  double density = 0.01;
  unsigned seed = 1234;

  // Supported flags:
  //   --M <int> --K <int> --N <int> --density <double> --seed <uint>
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--M") && i + 1 < argc) {
      M = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--K") && i + 1 < argc) {
      K = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--N") && i + 1 < argc) {
      N = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--density") && i + 1 < argc) {
      density = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
      seed = (unsigned)std::strtoul(argv[++i], nullptr, 10);
    } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      std::cout << "Usage: " << argv[0]
                << " [--M int] [--K int] [--N int] [--density double] [--seed "
                   "uint]\n";
      return 0;
    }
  }

  std::vector<int> row_ptr, col_idx;
  std::vector<float> vals;
  generate_random_csr(M, K, density, row_ptr, col_idx, vals, seed);
  int nnz = row_ptr.back();
  std::cout << "nnz = " << nnz << "\n";

  // Create B
  std::vector<float> B((size_t)K * N);
  for (size_t i = 0; i < B.size(); i++)
    B[i] = float(rand()) / RAND_MAX;

  // CPU reference
  std::vector<float> C_ref;
  spmm_cpu(M, K, N, row_ptr, col_idx, vals, B, C_ref);

  // Copy to device
  int *d_row_ptr, *d_col_idx;
  float *d_vals, *d_B, *d_C;
  cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
  cudaMalloc(&d_col_idx, nnz * sizeof(int));
  cudaMalloc(&d_vals, nnz * sizeof(float));
  cudaMalloc(&d_B, (size_t)K * N * sizeof(float));
  cudaMalloc(&d_C, (size_t)M * N * sizeof(float));

  cudaMemcpy(d_row_ptr, row_ptr.data(), (M + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_vals, vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), (size_t)K * N * sizeof(float),
             cudaMemcpyHostToDevice);

  // Launch incomplete student kernel
  int block = 256;
  int grid = (M + block - 1) / block;

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  cudaEventRecord(ev_start);
  spmm_csr_row_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B,
                                       d_C);
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  cudaDeviceSynchronize();

  // Copy back
  std::vector<float> C((size_t)M * N);
  cudaMemcpy(C.data(), d_C, (size_t)M * N * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Compare (will be wrong until students complete TODOs)
  float err = max_abs_err(C_ref, C);
  std::cout << "Max error = " << err << "\n";

  // FLOPs for SpMM: for each nonzero, N multiply-adds => 2 * nnz * N FLOPs
  double flops = 2.0 * (double)nnz * (double)N;
  double gflops =
      flops / ((double)ms * 1.0e6); // since GFLOP/s = FLOPs / (ms*1e6)
  std::cout << "PERF"
            << " M=" << M << " K=" << K << " N=" << N << " density=" << density
            << " nnz=" << nnz << " time_ms=" << ms << " gflops=" << gflops
            << "\n";

  cudaFree(d_row_ptr);
  cudaFree(d_col_idx);
  cudaFree(d_vals);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
