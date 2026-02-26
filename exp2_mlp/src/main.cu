#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlp_layers.cuh"

struct Options {
  std::vector<int> layers = {1024, 2048,
                             1024}; // includes input dim and final output dim
  int batch = 128;
  std::string activation = "relu";
  std::string impl = "baseline"; // baseline | activation_fused
  bool verify = true;
};

std::vector<int> parse_layers_list(const std::string &csv) {
  std::vector<int> dims;
  size_t start = 0;
  while (start < csv.size()) {
    size_t comma = csv.find(',', start);
    const size_t len =
        (comma == std::string::npos) ? (csv.size() - start) : (comma - start);
    if (len > 0) {
      dims.push_back(std::stoi(csv.substr(start, len)));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return dims;
}

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
      opt.layers = parse_layers_list(argv[++i]);
    } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
      opt.batch = std::stoi(argv[++i]);
    } else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) {
      opt.activation = argv[++i];
    } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
      opt.impl = argv[++i];
    } else if (strcmp(argv[i], "--no-verify") == 0) {
      opt.verify = false;
    } else if (strcmp(argv[i], "--help") == 0) {
      std::cout
          << "Usage: ./dmlp --layers 1024,2048,1024 --batch 128 --activation "
             "relu \\\n  --impl baseline|activation_fused [--no-verify]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
    }
  }
  if (opt.layers.size() < 2) {
    throw std::invalid_argument(
        "--layers must contain at least two integers (input/output)");
  }
  return opt;
}

void check_cuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + " : " +
                             cudaGetErrorString(err));
  }
}

void check_cublas(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + " : cuBLAS error");
  }
}

void seed_tensor(std::vector<float> &data, float scale) {
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = scale * std::sin(0.11f * static_cast<float>(i));
  }
}

static inline float gelu_cpu(float x) {
  // tanh approximation GELU
  const float kAlpha = 0.797885f; // sqrt(2/pi)
  const float kBeta = 0.044715f;
  return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + kBeta * x * x * x)));
}

// output = input * W^T
// input:  [B, in] row-major
// weight: [out, in] row-major (per assignment)
// output: [B, out] row-major
__inline__ void cpu_gemm_layer(const std::vector<float> &input,
                               const std::vector<float> &weights,
                               size_t w_offset, std::vector<float> &output,
                               const LayerShape &shape) {
  const int B = shape.batch;
  const int in = shape.in_dim;
  const int out = shape.out_dim;

  // Ensure output size
  output.assign(static_cast<size_t>(B) * out, 0.0f);

  // For each (b, j): sum over k
  for (int b = 0; b < B; ++b) {
    const size_t x_row = static_cast<size_t>(b) * in;
    const size_t y_row = static_cast<size_t>(b) * out;
    for (int j = 0; j < out; ++j) {
      const size_t w_row = w_offset + static_cast<size_t>(j) * in; // W[j, :]
      float acc = 0.0f;
      for (int k = 0; k < in; ++k) {
        acc += input[x_row + k] * weights[w_row + k];
      }
      output[y_row + j] = acc;
    }
  }
}

__inline__ void bias_add(const std::vector<float> &biases, size_t b_offset,
                         std::vector<float> &activations,
                         const LayerShape &shape) {
  const int B = shape.batch;
  const int out = shape.out_dim;

  for (int b = 0; b < B; ++b) {
    const size_t row = static_cast<size_t>(b) * out;
    for (int j = 0; j < out; ++j) {
      activations[row + j] += biases[b_offset + static_cast<size_t>(j)];
    }
  }
}

__inline__ void activation(std::string_view act,
                           std::vector<float> &activations,
                           const LayerShape &shape) {
  const size_t elements =
      static_cast<size_t>(shape.batch) * static_cast<size_t>(shape.out_dim);

  if (act == "relu") {
    for (size_t i = 0; i < elements; ++i) {
      if (activations[i] < 0.0f)
        activations[i] = 0.0f;
    }
  } else if (act == "gelu") {
    for (size_t i = 0; i < elements; ++i) {
      activations[i] = gelu_cpu(activations[i]);
    }
  } else {
    // If you add more activations, handle them here.
  }
}

void mlp_cpu_reference(const std::vector<int> &layers, int batch,
                       const std::vector<float> &weights,
                       const std::vector<float> &biases,
                       const std::vector<size_t> &weight_offsets,
                       const std::vector<size_t> &bias_offsets,
                       const std::vector<float> &input,
                       std::vector<float> &output, const std::string &act) {
  const int num_layers = static_cast<int>(layers.size()) - 1;

  std::vector<float> curr = input;
  std::vector<float> next;

  for (int layer = 0; layer < num_layers; ++layer) {
    LayerShape shape{batch, layers[layer], layers[layer + 1]};

    cpu_gemm_layer(curr, weights, weight_offsets[layer], next, shape);
    bias_add(biases, bias_offsets[layer], next, shape);
    activation(act, next, shape);

    curr.swap(next);
  }

  output = curr; // final activations
}

__inline__ float calculate_max_abs_diff(const std::vector<float> &a,
                                        const std::vector<float> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument(
        "Solution and Reference vectors must have the same size.");
  }

  float max_diff = 0.0f;
  for (size_t i = 0; i < a.size(); i++) {
    float diff = std::abs(a[i] - b[i]);
    if (diff > max_diff)
      max_diff = diff;
  }

  return max_diff;
}

int main(int argc, char **argv) {
  // Manage argument options
  Options opt = parse_args(argc, argv);
  const int batch = opt.batch;
  const size_t input_elems = static_cast<size_t>(batch) * opt.layers.front();
  const size_t output_elems = static_cast<size_t>(batch) * opt.layers.back();
  const int num_layers = static_cast<int>(opt.layers.size()) - 1;
  const int max_layer_width =
      *std::max_element(opt.layers.begin(), opt.layers.end());

  // Calculate offsets
  std::vector<size_t> weight_offsets(num_layers, 0);
  std::vector<size_t> bias_offsets(num_layers, 0);
  size_t weight_cursor = 0;
  size_t bias_cursor = 0;
  for (int i = 0; i < num_layers; ++i) {
    const int in_dim = opt.layers[i];
    const int out_dim = opt.layers[i + 1];
    weight_offsets[i] = weight_cursor;
    bias_offsets[i] = bias_cursor;
    weight_cursor += static_cast<size_t>(out_dim) * in_dim;
    bias_cursor += static_cast<size_t>(out_dim);
  }

  // Allocate host vectors
  std::vector<float> h_input(input_elems);
  std::vector<float> h_weights(weight_cursor);
  std::vector<float> h_biases(bias_cursor);
  std::vector<float> h_output(output_elems, 0.0f);
  std::vector<float> h_ref(output_elems, 0.0f);

  // Calculate the number of bytes needed for each device vector
  const size_t bytes_input = static_cast<size_t>(input_elems) * sizeof(float);
  const size_t bytes_weights =
      static_cast<size_t>(weight_cursor) * sizeof(float);
  const size_t bytes_biases = static_cast<size_t>(bias_cursor) * sizeof(float);
  const size_t bytes_output = static_cast<size_t>(output_elems) * sizeof(float);
  const size_t bytes_max_width =
      static_cast<size_t>(max_layer_width) * sizeof(float);

  // Fill values with random values
  seed_tensor(h_input, 1.0f);
  seed_tensor(h_weights, 0.25f);
  seed_tensor(h_biases, 0.01f);

  // Define ptrs to device vectors
  float *d_input = nullptr;
  float *d_workspace_a = nullptr;
  float *d_workspace_b = nullptr;
  float *d_weights = nullptr;
  float *d_biases = nullptr;

  // Allocate device buffers (activations + weights + biases)
  check_cuda(cudaMalloc((void **)&d_input, bytes_input),
             "allocate input array");
  check_cuda(cudaMalloc((void **)&d_weights, bytes_weights),
             "allocate weights array");
  check_cuda(cudaMalloc((void **)&d_biases, bytes_biases),
             "allocate biases array");
  check_cuda(cudaMalloc((void **)&d_workspace_a, bytes_max_width * batch),
             "allocate first workspace array");
  check_cuda(cudaMalloc((void **)&d_workspace_b, bytes_max_width * batch),
             "allocate second workspace array");

  // Copy Host data
  check_cuda(
      cudaMemcpy(d_input, h_input.data(), bytes_input, cudaMemcpyHostToDevice),
      "Copy input from host to device");
  check_cuda(cudaMemcpy(d_weights, h_weights.data(), bytes_weights,
                        cudaMemcpyHostToDevice),
             "Copy weights from host to device");
  check_cuda(cudaMemcpy(d_biases, h_biases.data(), bytes_biases,
                        cudaMemcpyHostToDevice),
             "Copy biases from host to device");

  // Copy Input Data to Workspace
  check_cuda(
      cudaMemcpy(d_workspace_a, d_input, bytes_input, cudaMemcpyDeviceToDevice),
      "Copy input from device to device");

  // Create events and streams
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");
  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream), "create stream");

  // Define handle
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

  float elapsed_ms = 0.0f;
  if (opt.impl == "baseline") {
    check_cuda(cudaEventRecord(start, stream), "record baseline start");
    for (int layer = 0; layer < num_layers; ++layer) {
      LayerShape shape{batch, opt.layers[layer], opt.layers[layer + 1]};
      const float *d_w = d_weights + weight_offsets[layer];
      const float *d_b = d_biases + bias_offsets[layer];
      check_cublas(
          run_gemm_layer(d_workspace_a, d_w, d_workspace_b, shape, handle),
          "run gemm layer");
      launch_bias_add(d_b, d_workspace_b, shape, stream);
      check_cuda(cudaGetLastError(), "kernel launch bias");
      check_cuda(cudaStreamSynchronize(stream), "kernel execution bias");
      launch_activation(opt.activation, d_workspace_b, shape, stream);
      check_cuda(cudaGetLastError(), "kernel launch bias");
      check_cuda(cudaStreamSynchronize(stream), "kernel execution bias");
      std::swap(d_workspace_a, d_workspace_b);
    }
    check_cuda(cudaEventRecord(stop, stream), "record baseline stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "elapsed baseline");
  } else if (opt.impl == "activation_fused") {
    check_cuda(cudaEventRecord(start, stream), "record fused start");
    for (int layer = 0; layer < num_layers; ++layer) {
      LayerShape shape{batch, opt.layers[layer], opt.layers[layer + 1]};
      const float *d_w = d_weights + weight_offsets[layer];
      const float *d_b = d_biases + bias_offsets[layer];
      run_gemm_layer(d_workspace_a, d_w, d_workspace_b, shape, handle);
      launch_fused_bias_activation(d_b, opt.activation, d_workspace_b, shape,
                                   stream);
      std::swap(d_workspace_a, d_workspace_b);
    }
    check_cuda(cudaEventRecord(stop, stream), "record fused stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed fused");
  } else {
    throw std::invalid_argument("Unknown --impl " + opt.impl);
  }

  // copy final activations back to h_output.
  check_cuda(cudaMemcpy(h_output.data(), d_workspace_a, bytes_output,
                        cudaMemcpyDeviceToHost),
             "Copy output from device to host");

  if (opt.verify) {
    mlp_cpu_reference(opt.layers, batch, h_weights, h_biases, weight_offsets,
                      bias_offsets, h_input, h_ref, opt.activation);
    float max_diff = calculate_max_abs_diff(h_output, h_ref);
    std::cout << "Maximum Difference: " << max_diff << std::endl;
  }

  if (elapsed_ms > 0.0f) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Impl=" << opt.impl << " Batch=" << batch << " Layers=";
    for (size_t i = 0; i < opt.layers.size(); ++i) {
      std::cout << opt.layers[i];
      if (i + 1 < opt.layers.size()) {
        std::cout << "x";
      }
    }
    std::cout << " Time(ms)=" << elapsed_ms
              << " GFLOP/s=" << mlp_gflops(opt.layers, batch, elapsed_ms)
              << std::endl;
  } else {
    std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
  }

  // cleanup (cudaFree buffers, destroy events/stream/handle).
  check_cuda(cudaFree(d_input), "Freeing device input vector");
  check_cuda(cudaFree(d_biases), "Freeing device bias vector");
  check_cuda(cudaFree(d_weights), "Freeing device weights vector");
  check_cuda(cudaFree(d_workspace_a), "Freeing first workspace");
  check_cuda(cudaFree(d_workspace_b), "Freeing second workspace");

  check_cublas(cublasDestroy(handle), "Destroying cublas handle");

  check_cuda(cudaStreamDestroy(stream), "Destroying stream");
  check_cuda(cudaEventDestroy(start), "Destroying start event");
  check_cuda(cudaEventDestroy(stop), "Destroying stop event");

  return 0;
}
