#pragma once

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <vector>

struct LayerShape {
  int batch;
  int in_dim;
  int out_dim;
};

inline double layer_flops(const LayerShape &shape) {
  return 2.0 * static_cast<double>(shape.batch) * shape.in_dim * shape.out_dim;
}

inline double mlp_gflops(const std::vector<int> &layers, int batch,
                         double millis) {
  double total_flops = 0.0;
  for (size_t i = 0; i + 1 < layers.size(); ++i) {
    LayerShape shape{batch, layers[i], layers[i + 1]};
    total_flops += layer_flops(shape);
  }
  return total_flops / (millis * 1e6);
}

__device__ __forceinline__ float gelu(float x) {
  const float kAlpha = 0.797885f; // sqrt(2/pi)
  const float kBeta = 0.044715f;

  return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + kBeta * x * x * x)));
}

__global__ void bias_add_kernel(const float *__restrict__ bias,
                                float *__restrict__ activations,
                                LayerShape shape) {
  int batch = shape.batch;
  int out_dim = shape.out_dim;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elements = batch * out_dim;

  if (idx >= elements)
    return;

  int bias_idx = idx % out_dim;
  activations[idx] += bias[bias_idx];
}

__global__ void relu_kernel(float *__restrict__ activations, size_t elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements)
    return;

  float activation = activations[idx];
  if (activation < 0.0f)
    activations[idx] = 0.0f;
}

__global__ void gelu_kernel(float *__restrict__ activations, size_t elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements)
    return;

  activations[idx] = gelu(activations[idx]);
}

inline void launch_bias_add(const float *bias, float *activations,
                            const LayerShape &shape, cudaStream_t stream) {
  const int threads = 256;
  const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  bias_add_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape);
}

inline void launch_activation(const std::string &activation, float *activations,
                              const LayerShape &shape, cudaStream_t stream) {
  const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  if (activation == "relu") {
    relu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
  } else if (activation == "gelu") {
    gelu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
  }
}

__global__ void fused_bias_activation_kernel(const float *__restrict__ bias,
                                             float *__restrict__ activations,
                                             LayerShape shape,
                                             int activation_type) {
  int batch = shape.batch;
  int out_dim = shape.out_dim;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elements = batch * out_dim;

  if (idx >= elements)
    return;

  int bias_idx = idx % out_dim;

  float v = activations[idx];
  v += bias[bias_idx];

  if (activation_type == 0) { // relu
    if (v < 0.0f)
      v = 0.0f;
  } else if (activation_type == 1) { // gelu
    v = gelu(v);
  }

  activations[idx] = v;
}

inline void launch_fused_bias_activation(const float *bias,
                                         const std::string &activation,
                                         float *activations,
                                         const LayerShape &shape,
                                         cudaStream_t stream) {
  int activation_type = (activation == "gelu") ? 1 : 0;
  const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  fused_bias_activation_kernel<<<blocks, threads, 0, stream>>>(
      bias, activations, shape, activation_type);
  (void)elements;
}

inline void run_gemm_layer(const float *input, const float *weight,
                           float *output, const LayerShape &shape,
                           cublasHandle_t handle) {
  const int B = shape.batch;
  const int in = shape.in_dim;
  const int out = shape.out_dim;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int lda = in;
  const int ldb = in;
  const int ldc = out;

  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out, B, in, &alpha, weight, lda,
              input, ldb, &beta, output, ldc);
}
