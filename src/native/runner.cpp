#include <cuda.h>
#include <cublasLt.h>
#include <cudnn.h>

#define GATERV3_BUILD
#include "runner.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr int tile = 128;
constexpr size_t rgb_elements = 3 * tile * tile;
constexpr size_t scratch_bytes = 64 * 1024 * 1024;
constexpr int threads = 256;
constexpr float one = 1, zero = 0;
constexpr uint16_t half_one = 0x3c00, half_zero = 0;
thread_local std::string error_text;

void check(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* detail = nullptr;
    cuGetErrorString(status, &detail);
    throw std::runtime_error(std::string(operation) + ": " + (detail ? detail : "CUDA error"));
}

void check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
}

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
}

#define CHECK(expr) check((expr), #expr)

struct ContextScope {
    bool pushed = false;
    explicit ContextScope(CUcontext context) {
        CUcontext current;
        CHECK(cuCtxGetCurrent(&current));
        if (current != context) {
            CHECK(cuCtxPushCurrent(context));
            pushed = true;
        }
    }
    ~ContextScope() {
        if (pushed) {
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    }
};

enum Kind {
    Pack, Conv, Rms, Local, MeanProject, LocalSkip, Inception, GlobalSkip, Shuffle,
    Concat, Silu, Span, Depthwise, Attention, Gate, Add, Rgb, ShuffleConcat,
};

struct Tensor {
    int h, w, c, offset;
    int pixels() const { return h * w; }
    int elements() const { return pixels() * c; }
};

struct Op {
    int kind, out;
    std::array<int, 4> input;
    std::array<int, 3> weight;
    int x, y;
};
static_assert(sizeof(Op) == 44 && sizeof(Tensor) == 16);

void read(std::ifstream& file, void* target, size_t bytes) {
    if (!file.read(static_cast<char*>(target), bytes))
        throw std::runtime_error("Could not read native artifact");
}

struct Convolution {
    cudnnTensorDescriptor_t input = nullptr, output = nullptr;
    cudnnFilterDescriptor_t weight = nullptr;
    cudnnConvolutionDescriptor_t conv = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm{};
    size_t workspace = 0;

    ~Convolution() {
        if (input) cudnnDestroyTensorDescriptor(input);
        if (output) cudnnDestroyTensorDescriptor(output);
        if (weight) cudnnDestroyFilterDescriptor(weight);
        if (conv) cudnnDestroyConvolutionDescriptor(conv);
    }
};

struct Matmul {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t kernel = nullptr, input = nullptr, output = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace = 0;

    ~Matmul() {
        if (desc) cublasLtMatmulDescDestroy(desc);
        if (kernel) cublasLtMatrixLayoutDestroy(kernel);
        if (input) cublasLtMatrixLayoutDestroy(input);
        if (output) cublasLtMatrixLayoutDestroy(output);
    }
};

struct GemmChoice {
    cublasLtMatmulAlgo_t algorithm;
    size_t workspace;
};

struct TimedGraph {
    std::vector<CUevent> events;
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    explicit TimedGraph(size_t count) : events(count, nullptr) {}
    ~TimedGraph() {
        if (executable) cuGraphExecDestroy(executable);
        if (graph) cuGraphDestroy(graph);
        for (auto event : events) if (event) cuEventDestroy(event);
    }
};

struct TuningTimer {
    CUevent start = nullptr, stop = nullptr;
    ~TuningTimer() {
        if (start) cuEventDestroy(start);
        if (stop) cuEventDestroy(stop);
    }
    void initialize() {
        CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
        CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    }
    template<class Launch>
    float measure(CUstream stream, Launch launch) {
        if (!launch()) return 1e30f;
        CHECK(cuEventRecord(start, stream));
        for (int i = 0; i < 5; ++i) {
            if (!launch()) return 1e30f;
        }
        CHECK(cuEventRecord(stop, stream));
        CHECK(cuEventSynchronize(stop));
        float elapsed;
        CHECK(cuEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }
};

} // namespace

class GtRunner {
public:
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUmodule module = nullptr;
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    CUdeviceptr input = 0, output = 0, weights = 0, arena = 0, scratch = 0;
    void* host_input = nullptr;
    void* host_output = nullptr;
    cudnnHandle_t cudnn = nullptr;
    cublasLtHandle_t blas = nullptr;
    int batch = 0;
    bool fp16 = false;
    size_t io_bytes = 0;

    void buffer_sizes(uint64_t bytes[4]) const {
        bytes[0] = weight_bytes;
        bytes[1] = arena_bytes;
        bytes[2] = workspace_bytes;
        bytes[3] = 2 * io_bytes * batch;
    }

    void startup_memory(uint64_t bytes[4]) const {
        std::copy(startup_used.begin(), startup_used.end(), bytes);
    }

    void initialize(const fs::path& directory, int size, bool capture, bool half_precision) {
        if (size != 1 && size != 16) throw std::runtime_error("Supported batches are 1 and 16");
        batch = size;
        fp16 = half_precision;
        io_bytes = rgb_elements * (fp16 ? sizeof(uint16_t) : sizeof(float));
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, 0));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        ContextScope scope(context);
        record_memory(0);
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuModuleLoad(&module, (directory / "kernels.cubin").u8string().c_str()));
        CHECK(cudnnCreate(&cudnn));
        CHECK(cublasLtCreate(&blas));
        CHECK(cudnnSetStream(cudnn, stream));
        load(directory);
        CHECK(cuMemAlloc(&scratch, scratch_bytes));
        record_memory(1);
        CHECK(cuMemAllocHost(&host_input, io_bytes * batch));
        CHECK(cuMemAllocHost(&host_output, io_bytes * batch));
        CHECK(cuMemsetD8Async(arena, 0x28, arena_bytes, stream));
        prepare();
        CHECK(cuStreamSynchronize(stream));
        record_memory(2);
        CHECK(cuMemFree(scratch));
        scratch = 0;
        CHECK(cuMemAlloc(&scratch, workspace_bytes));
        CHECK(cuMemsetD8Async(input, 0, io_bytes * batch, stream));
        forward(stream);
        CHECK(cuStreamSynchronize(stream));
        if (capture) {
            CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            try {
                forward(stream);
            } catch (...) {
                cuStreamEndCapture(stream, &graph);
                throw;
            }
            CHECK(cuStreamEndCapture(stream, &graph));
            CHECK(cuGraphInstantiate(&executable, graph, 0));
            CHECK(cuGraphUpload(executable, stream));
            CHECK(cuStreamSynchronize(stream));
            CHECK(cuGraphDestroy(graph));
            graph = nullptr;
        }
        record_memory(3);
    }

    ~GtRunner() {
        if (!context) return;
        // Destruction also works if initialization failed partway through.
        if (cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
        if (stream) cuStreamSynchronize(stream);
        if (executable) cuGraphExecDestroy(executable);
        if (graph) cuGraphDestroy(graph);
        matmuls.clear();
        convolutions.clear();
        if (cudnn) cudnnDestroy(cudnn);
        if (blas) cublasLtDestroy(blas);
        if (module) cuModuleUnload(module);
        for (auto ptr : {weights, arena, scratch, input, output}) if (ptr) cuMemFree(ptr);
        if (host_input) cuMemFreeHost(host_input);
        if (host_output) cuMemFreeHost(host_output);
        if (stream) cuStreamDestroy(stream);
        CUcontext previous;
        cuCtxPopCurrent(&previous);
        cuDevicePrimaryCtxRelease(device);
    }

    void enqueue(CUstream target) {
        if (executable) {
            CHECK(cuGraphLaunch(executable, target));
        } else {
            CHECK(cudnnSetStream(cudnn, target));
            forward(target);
        }
    }

    void run_host(const void* pixels, void* restored) {
        ContextScope scope(context);
        std::memcpy(host_input, pixels, batch * io_bytes);
        CHECK(cuMemcpyHtoDAsync(input, host_input, batch * io_bytes, stream));
        enqueue(stream);
        CHECK(cuMemcpyDtoHAsync(host_output, output, batch * io_bytes, stream));
        CHECK(cuStreamSynchronize(stream));
        std::memcpy(restored, host_output, batch * io_bytes);
    }

    int operation_count() const { return int(ops.size()); }

    void profile(float* milliseconds, int count) {
        if (count != operation_count()) throw std::runtime_error("Invalid profile buffer size");
        ContextScope scope(context);
        TimedGraph timed(ops.size() + 1);
        for (auto& event : timed.events) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        CHECK(cudnnSetStream(cudnn, stream));
        CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        try {
            forward(stream, timed.events.data());
        } catch (...) {
            cuStreamEndCapture(stream, &timed.graph);
            throw;
        }
        CHECK(cuStreamEndCapture(stream, &timed.graph));
        CHECK(cuGraphInstantiate(&timed.executable, timed.graph, 0));
        constexpr int iterations = 8;
        for (int run = 0; run <= iterations; ++run) {
            CHECK(cuGraphLaunch(timed.executable, stream));
            CHECK(cuStreamSynchronize(stream));
            if (!run) continue;
            for (int i = 0; i < count; ++i) {
                float elapsed;
                CHECK(cuEventElapsedTime(&elapsed, timed.events[i], timed.events[i + 1]));
                milliseconds[i] += elapsed / iterations;
            }
        }
    }

private:
    std::vector<Tensor> tensors;
    std::vector<Op> ops;
    std::vector<CUdeviceptr> addresses;
    std::vector<Convolution*> conv_by_op;
    std::vector<std::unique_ptr<Matmul>> matmuls;
    std::map<std::array<int, 5>, GemmChoice> gemm_algorithms;
    std::map<std::array<int, 5>, std::unique_ptr<Convolution>> convolutions;
    std::map<std::string, CUfunction> kernels;
    std::vector<CUfunction> kernel_by_op;
    CUfunction add_bias = nullptr;
    CUfunction mean_project = nullptr;
    size_t arena_bytes = 0;
    size_t weight_bytes = 0;
    size_t workspace_bytes = 0;
    std::array<uint64_t, 4> startup_used{};

    void record_memory(size_t stage) {
        size_t available, total;
        CHECK(cuMemGetInfo(&available, &total));
        startup_used[stage] = total - available;
    }

    void load(const fs::path& directory) {
        std::ifstream schedule(directory / "schedule.bin", std::ios::binary);
        std::array<uint32_t, 4> header{};
        read(schedule, header.data(), sizeof(header));
        bool legacy = header[0] == 0x32525447;
        if ((!legacy && header[0] != 0x33525447) || (legacy && fp16) || header[1] != header[2] + 1
            || header[2] < 2 || header[2] > 2000 || header[3] > (256u << 20))
            throw std::runtime_error("Invalid GateRV3 schedule header");
        tensors.resize(header[1]);
        ops.resize(header[2]);
        read(schedule, tensors.data(), tensors.size() * sizeof(Tensor));
        read(schedule, ops.data(), ops.size() * sizeof(Op));
        if (tensors.front().offset != -1 || tensors.back().offset != -1
            || tensors.front().elements() != 3 * tile * tile
            || tensors.back().elements() != 3 * tile * tile)
            throw std::runtime_error("Invalid native input or output shape");
        size_t extent = 0;
        for (size_t i = 1; i + 1 < tensors.size(); ++i) {
            const auto& t = tensors[i];
            if (t.h < 1 || t.h > tile || t.w < 1 || t.w > tile
                || t.c < 1 || t.c > 2560 || t.offset < 0 || t.offset % 256)
                throw std::runtime_error("Invalid native tensor");
            extent = std::max(extent, size_t(t.offset) + t.elements() * 2);
        }
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& op = ops[i];
            if (op.kind < Pack || op.kind > (legacy ? Rgb : ShuffleConcat) || op.out != i + 1
                || op.input[0] < 0
                || std::any_of(op.input.begin(), op.input.end(), [i](int id) { return id < -1 || id > int(i); })
                || std::any_of(op.weight.begin(), op.weight.end(), [&](int id) {
                    return id < -1 || (id >= 0 && (id % 256 || uint32_t(id) >= header[3]));
                }))
                throw std::runtime_error("Invalid native operation");
        }
        if (extent > (32u << 20) || ops.front().kind != Pack || ops.back().kind != Rgb
            || (fp16 && ops.back().weight[1] < 0))
            throw std::runtime_error("Invalid native schedule");
        std::vector<char> staging(header[3]);
        weight_bytes = staging.size();
        std::ifstream file(directory / "weights.bin", std::ios::binary);
        read(file, staging.data(), staging.size());
        if (file.peek() != EOF) throw std::runtime_error("Unexpected trailing weight data");
        CHECK(cuMemAlloc(&weights, staging.size()));
        CHECK(cuMemcpyHtoD(weights, staging.data(), staging.size()));
        CHECK(cuMemAlloc(&input, batch * io_bytes));
        CHECK(cuMemAlloc(&output, batch * io_bytes));
        arena_bytes = extent * batch;
        CHECK(cuMemAlloc(&arena, arena_bytes));
        addresses.resize(tensors.size());
        addresses.front() = input;
        addresses.back() = output;
        for (size_t i = 1; i + 1 < tensors.size(); ++i)
            addresses[i] = arena + size_t(tensors[i].offset) * batch;
    }

    CUfunction lookup(std::string name) {
        auto [it, created] = kernels.try_emplace(std::move(name));
        if (created) CHECK(cuModuleGetFunction(&it->second, module, it->first.c_str()));
        return it->second;
    }

    CUdeviceptr weight(int offset) const { return offset < 0 ? 0 : weights + offset; }
    CUdeviceptr tensor(int id) const { return id < 0 ? 0 : addresses[id]; }
    int channels(int id) const { return id < 0 ? 0 : tensors[id].c; }
    bool fused_norm(size_t index) const {
        return index < ops.size() && ops[index].kind == Rms && ops[index].input[1] >= 0;
    }

    cublasStatus_t gemm(const Matmul& p, const Op& op, const cublasLtMatmulAlgo_t& algo,
                        size_t workspace, CUstream target) {
        auto y = pointer(tensor(op.out));
        return cublasLtMatmul(blas, p.desc, fp16 ? static_cast<const void*>(&half_one) : &one,
            pointer(weight(op.weight[0])), p.kernel, pointer(tensor(op.input[0]) + 2 * op.y),
            p.input, fp16 ? static_cast<const void*>(&half_zero) : &zero, y, p.output,
            y, p.output, &algo, workspace ? pointer(scratch) : nullptr, workspace, target);
    }

    void prepare_matmul(size_t index, const Tensor& a, const Tensor& out, const Op& op, TuningTimer& timer) {
        auto plan = std::make_unique<Matmul>();
        CHECK(cublasLtMatmulDescCreate(&plan->desc, fp16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
                                      fp16 ? CUDA_R_16F : CUDA_R_32F));
        // NHWC pixels are columns, with channels contiguous within each column.
        // The packed weights have one contiguous row per output channel.
        cublasOperation_t transpose = CUBLAS_OP_T;
        CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transpose, sizeof(transpose)));
        int cin = a.c - op.y, count = out.pixels() * batch;
        CHECK(cublasLtMatrixLayoutCreate(&plan->kernel, CUDA_R_16F, cin, out.c, cin));
        CHECK(cublasLtMatrixLayoutCreate(&plan->input, CUDA_R_16F, cin, count, a.c));
        CHECK(cublasLtMatrixLayoutCreate(&plan->output, CUDA_R_16F, out.c, count, out.c));
        if (op.weight[1] >= 0) {
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            CUdeviceptr ptr = weight(op.weight[1]);
            CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
            CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &ptr, sizeof(ptr)));
        }
        auto key = std::array<int, 5>{cin, out.c, count, a.c, op.weight[1] >= 0};
        if (auto cached = gemm_algorithms.find(key); cached != gemm_algorithms.end()) {
            plan->algorithm = cached->second.algorithm;
            plan->workspace = cached->second.workspace;
            matmuls[index] = std::move(plan);
            return;
        }
        cublasLtMatmulPreference_t pref;
        CHECK(cublasLtMatmulPreferenceCreate(&pref));
        auto preference = std::unique_ptr<std::remove_pointer_t<cublasLtMatmulPreference_t>,
            decltype(&cublasLtMatmulPreferenceDestroy)>(pref, cublasLtMatmulPreferenceDestroy);
        uint64_t allotted = scratch_bytes;
        CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &allotted, sizeof(allotted)));
        std::array<cublasLtMatmulHeuristicResult_t, 8> results{};
        int found = 0;
        CHECK(cublasLtMatmulAlgoGetHeuristic(blas, plan->desc, plan->kernel, plan->input,
                                            plan->output, plan->output, pref,
                                            int(results.size()), results.data(), &found));
        float best = 1e30f;
        for (int j = 0; j < found; ++j) {
            auto& candidate = results[j];
            if (candidate.state != CUBLAS_STATUS_SUCCESS) continue;
            float elapsed = timer.measure(stream, [&] {
                return gemm(*plan, op, candidate.algo, candidate.workspaceSize, stream) == CUBLAS_STATUS_SUCCESS;
            });
            if (elapsed < best) {
                best = elapsed;
                plan->algorithm = candidate.algo;
                plan->workspace = candidate.workspaceSize;
            }
        }
        if (best == 1e30f) throw std::runtime_error("No working cuBLASLt plan for operation " + std::to_string(index));
        workspace_bytes = std::max(workspace_bytes, plan->workspace);
        gemm_algorithms.emplace(key, GemmChoice{plan->algorithm, plan->workspace});
        matmuls[index] = std::move(plan);
    }

    cudnnStatus_t convolution(const Convolution& p, const Op& op) {
        return cudnnConvolutionForward(cudnn, &one, p.input, pointer(tensor(op.input[0])),
            p.weight, pointer(weight(op.weight[0])), p.conv, p.algorithm, pointer(scratch),
            p.workspace, &zero, p.output, pointer(tensor(op.out)));
    }

    void prepare_conv(size_t index, const Tensor& in, const Tensor& out, const Op& op, TuningTimer& timer) {
        auto key = std::array<int, 5>{in.h, in.w, in.c, out.c, op.x};
        auto [it, inserted] = convolutions.try_emplace(key);
        if (inserted) {
            it->second = std::make_unique<Convolution>();
            auto& p = *it->second;
            CHECK(cudnnCreateTensorDescriptor(&p.input));
            CHECK(cudnnCreateTensorDescriptor(&p.output));
            CHECK(cudnnCreateFilterDescriptor(&p.weight));
            CHECK(cudnnCreateConvolutionDescriptor(&p.conv));
            CHECK(cudnnSetTensor4dDescriptor(p.input, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
                                             batch, in.c, in.h, in.w));
            CHECK(cudnnSetTensor4dDescriptor(p.output, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
                                             batch, out.c, out.h, out.w));
            CHECK(cudnnSetFilter4dDescriptor(p.weight, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC,
                                            out.c, in.c, op.x, op.x));
            CHECK(cudnnSetConvolution2dDescriptor(p.conv, op.x / 2, op.x / 2,
                1, 1, 1, 1, CUDNN_CROSS_CORRELATION, fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT));
            CHECK(cudnnSetConvolutionMathType(p.conv, CUDNN_TENSOR_OP_MATH));
            std::array<cudnnConvolutionFwdAlgoPerf_t, 16> results;
            int count = 0;
            CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, p.input, p.weight, p.conv,
                                                        p.output, int(results.size()), &count, results.data()));
            const cudnnConvolutionFwdAlgoPerf_t* fastest = nullptr;
            float best = 1e30f;
            for (int j = 0; j < count; ++j) {
                const auto& candidate = results[j];
                if (candidate.status != CUDNN_STATUS_SUCCESS || candidate.memory > scratch_bytes) continue;
                CHECK(cudnnSetConvolutionMathType(p.conv, candidate.mathType));
                p.algorithm = candidate.algo;
                p.workspace = candidate.memory;
                float elapsed = timer.measure(stream, [&] {
                    return convolution(p, op) == CUDNN_STATUS_SUCCESS;
                });
                if (elapsed < best) {
                    fastest = &candidate;
                    best = elapsed;
                }
            }
            if (!fastest)
                throw std::runtime_error("No cuDNN plan for operation " + std::to_string(index));
            p.algorithm = fastest->algo;
            p.workspace = fastest->memory;
            workspace_bytes = std::max(workspace_bytes, p.workspace);
            CHECK(cudnnSetConvolutionMathType(p.conv, fastest->mathType));
        }
        conv_by_op[index] = it->second.get();
    }

    void prepare() {
        conv_by_op.resize(ops.size());
        matmuls.resize(ops.size());
        kernel_by_op.resize(ops.size());
        TuningTimer timer;
        timer.initialize();
        add_bias = lookup(fp16 ? "bias16" : "bias");
        mean_project = lookup(fp16 ? "mean_project16" : "mean_project");
        constexpr const char* names[] = {
            "pack", nullptr, nullptr, nullptr, "mean_partial", "local_skip",
            nullptr, "global_skip", "shuffle", "concat", "silu", "span_gate",
            "depthwise", "attention", "gate", "add", "rgb", "shuffle_concat",
        };
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& op = ops[i];
            const auto& out = tensors[op.out];
            if (op.kind == MeanProject) {
                const auto pixels = tensors[op.input[0]].pixels();
                size_t partial_bytes = size_t((pixels + 255) / 256) * batch * out.c
                    * (fp16 ? sizeof(uint16_t) : sizeof(float));
                workspace_bytes = std::max(workspace_bytes, partial_bytes);
            }
            if (op.kind == Conv) {
                const auto& in = tensors[op.input[0]];
                if (op.x == 1) prepare_matmul(i, in, out, op, timer);
                else prepare_conv(i, in, out, op, timer);
            } else if (op.kind == Rms) {
                std::string name = fp16 ? "rms16_" : "rms";
                if (fused_norm(i)) {
                    if (!i || (ops[i - 1].kind != LocalSkip && ops[i - 1].kind != GlobalSkip))
                        throw std::runtime_error("Invalid fused RMS operation");
                    name = ops[i - 1].kind == LocalSkip ? "local_norm" : "global_norm";
                    if (fp16) name += "16_";
                }
                kernel_by_op[i] = lookup(name + std::to_string(out.c));
            } else if (op.kind == Local) {
                kernel_by_op[i] = lookup((fp16 ? "local_gate16_" : "local_gate") + std::to_string(out.c));
            } else if (op.kind == Inception) {
                kernel_by_op[i] = lookup((fp16 ? "inception16_" : "inception") + std::to_string(2 * out.c / 3));
            } else {
                std::string name = names[op.kind];
                if (fp16 && op.kind != Shuffle && op.kind != Concat && op.kind != ShuffleConcat) name += "16";
                kernel_by_op[i] = lookup(name);
            }
        }
    }

    template<class... Args>
    void launch(CUfunction kernel, CUstream target, unsigned grid_x, unsigned grid_y,
                unsigned block, Args... args) {
        void* params[] = {static_cast<void*>(&args)...};
        CHECK(cuLaunchKernel(kernel, grid_x, grid_y, 1, block, 1, 1, 0, target, params, nullptr));
    }

    void forward(CUstream target, const CUevent* timing = nullptr) {
        for (size_t i = 0; i < ops.size(); ++i) {
            if (timing) CHECK(cuEventRecordWithFlags(timing[i], target, CU_EVENT_RECORD_EXTERNAL));
            const auto& op = ops[i];
            const auto& o = tensors[op.out];
            const auto& in = tensors[op.input[0]];
            const auto a = tensor(op.input[0]), b = tensor(op.input[1]);
            const auto c = tensor(op.input[2]), d = tensor(op.input[3]);
            const auto w0 = weight(op.weight[0]), w1 = weight(op.weight[1]);
            const auto y = addresses[op.out];
            const int total = o.elements() * batch, pixels = o.pixels();
            const unsigned grid = (total + threads - 1) / threads;
            auto kernel = kernel_by_op[i];
            const auto run = [&](auto... args) { launch(kernel, target, grid, 1, threads, args...); };
            const auto pairs = [&](auto... args) {
                launch(kernel, target, (total / 2 + threads - 1) / threads, 1, threads, args..., total / 2);
            };
            if ((op.kind == LocalSkip || op.kind == GlobalSkip) && fused_norm(i + 1)) {
                const auto& next = ops[i + 1];
                int per_block = o.c == 32 ? 16 : 8;
                launch(kernel_by_op[i + 1], target, (pixels * batch + per_block - 1) / per_block,
                    1, threads, y, tensor(next.out), a, b, c, w0, weight(next.weight[0]),
                    weight(next.weight[1]), pixels, pixels * batch);
                continue;
            }
            switch (op.kind) {
            case Pack: run(y, a, pixels, total); break;
            case Conv:
                if (auto* p = matmuls[i].get()) {
                    CHECK(gemm(*p, op, p->algorithm, p->workspace, target));
                } else {
                    CHECK(convolution(*conv_by_op[i], op));
                    if (w1) launch(add_bias, target, grid, 1, threads, y, w1, o.c, total);
                }
                break;
            case Rms: {
                if (fused_norm(i)) break;
                int per_block = o.c == 32 ? 16 : 8;
                launch(kernel, target, (pixels * batch + per_block - 1) / per_block, 1, threads,
                       y, a, w0, w1, pixels * batch);
                break;
            }
            case Local: pairs(y, a, w0, w1); break;
            case MeanProject: {
                int chunks = (in.pixels() + 255) / 256;
                launch(kernel, target, o.c / 32, batch * chunks, threads,
                       scratch, a, in.pixels(), o.c, chunks);
                launch(mean_project, target, batch, 1, threads, y, scratch, w0, w1, in.pixels(), o.c, chunks);
                break;
            }
            case LocalSkip: pairs(y, a, b, c, w0, pixels, o.c / 2); break;
            case Inception: pairs(y, a, w0, w1); break;
            case GlobalSkip: pairs(y, a, b, w0, o.c / 2); break;
            case Shuffle: run(y, a, o.h, o.w, o.c, op.x, total); break;
            case Concat:
                launch(kernel, target, (total / 8 + threads - 1) / threads, 1, threads, y, a, b, c, d,
                       in.c / 8, channels(op.input[1]) / 8, channels(op.input[2]) / 8,
                       channels(op.input[3]) / 8, o.c / 8, total / 8);
                break;
            case Silu: pairs(y, a); break;
            case Span: pairs(y, a, b); break;
            case Depthwise: run(y, a, w0, w1, in.c, total); break;
            case Attention: launch(kernel, target, batch * 16, 1, threads, y, a); break;
            case Gate: run(y, a, b, in.c, total); break;
            case Add: run(y, a, b, total); break;
            case Rgb: run(y, a, b, fp16 ? w1 : w0, pixels, total); break;
            case ShuffleConcat:
                pairs(y, a, b, o.h, o.w, o.c / 2);
                break;
            }
        }
        if (timing) CHECK(cuEventRecordWithFlags(timing[ops.size()], target, CU_EVENT_RECORD_EXTERNAL));
    }

    static void* pointer(CUdeviceptr p) { return reinterpret_cast<void*>(p); }
};

namespace {
template<class F>
int report_errors(F action) noexcept {
    try {
        action();
        return 0;
    } catch (const std::exception& error) {
        error_text = error.what();
        return -1;
    }
}
} // namespace

extern "C" {
const char* gt_error() { return error_text.c_str(); }

GtRunner* gt_create(const char* directory, int batch, int graph) {
    GtRunner* result = nullptr;
    report_errors([&] {
        auto runner = std::make_unique<GtRunner>();
        runner->initialize(fs::u8path(directory), batch, graph != 0, false);
        result = runner.release();
    });
    return result;
}

GtRunner* gt_create_fp16(const char* directory, int batch, int graph) {
    GtRunner* result = nullptr;
    report_errors([&] {
        auto runner = std::make_unique<GtRunner>();
        runner->initialize(fs::u8path(directory), batch, graph != 0, true);
        result = runner.release();
    });
    return result;
}

void gt_destroy(GtRunner* runner) { delete runner; }
uint64_t gt_input(const GtRunner* runner) { return runner->input; }
uint64_t gt_output(const GtRunner* runner) { return runner->output; }
int gt_operation_count(const GtRunner* runner) { return runner->operation_count(); }

int gt_profile(GtRunner* runner, float* milliseconds, int count) {
    return report_errors([&] { runner->profile(milliseconds, count); });
}

int gt_enqueue(GtRunner* runner, GtStream stream) {
    return report_errors([&] {
        ContextScope scope(runner->context);
        runner->enqueue(stream);
    });
}

int gt_run(GtRunner* runner, const float* input, float* output) {
    return report_errors([&] {
        if (runner->fp16) throw std::runtime_error("Use gt_run_fp16 for a full-FP16 runner");
        runner->run_host(input, output);
    });
}

int gt_run_fp16(GtRunner* runner, const uint16_t* input, uint16_t* output) {
    return report_errors([&] {
        if (!runner->fp16) throw std::runtime_error("Use gt_run for a mixed-precision runner");
        runner->run_host(input, output);
    });
}

void gt_buffer_sizes(const GtRunner* runner, uint64_t bytes[4]) {
    runner->buffer_sizes(bytes);
}

void gt_startup_memory(const GtRunner* runner, uint64_t bytes[4]) {
    runner->startup_memory(bytes);
}
}

int main(int argc, char** argv) {
    bool fp16 = argc == 6 && std::string(argv[5]) == "--fp16";
    if (argc != 5 && !fp16) {
        std::cerr << "Usage: gaterv3 <bundle-directory> <batch: 1|16> <input> <output> [--fp16]\n";
        return 2;
    }
    return report_errors([&] {
        auto create = fp16 ? gt_create_fp16 : gt_create;
        std::unique_ptr<GtRunner> runner(create(argv[1], std::stoi(argv[2]), 1));
        if (!runner) throw std::runtime_error(error_text);
        std::vector<char> pixels(runner->batch * runner->io_bytes);
        std::vector<char> restored(pixels.size());
        std::ifstream file(fs::u8path(argv[3]), std::ios::binary);
        read(file, pixels.data(), pixels.size());
        if (file.peek() != EOF) throw std::runtime_error("Input has extra data");
        runner->run_host(pixels.data(), restored.data());
        std::ofstream out(fs::u8path(argv[4]), std::ios::binary);
        if (!out.write(restored.data(), restored.size()))
            throw std::runtime_error("Could not write restored RGB output");
    }) == 0 ? 0 : (std::cerr << error_text << '\n', 1);
}
