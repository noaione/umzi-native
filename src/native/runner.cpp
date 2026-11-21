#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cuda.h>
#include <cublasLt.h>
#include <cudnn.h>

#define GATERV3_BUILD
#include "runner.h"
#include "plan_cache.h"

#include <algorithm>
#include <array>
#include <chrono>
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
using gt::ShapeKey;
using gt::GemmChoice;
using Clock = std::chrono::steady_clock;

double milliseconds_since(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

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

#include "silu_plan.h"

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
    bool fuse_silu = false;
    size_t io_bytes = 0;
    gt::PlanCache plan_cache;
    std::array<double, 4> setup_ms{};
    int fused_silu_count = 0;
    std::string fusion_diagnostic;

    void buffer_sizes(uint64_t bytes[4]) const {
        bytes[0] = weight_bytes;
        bytes[1] = arena_bytes;
        bytes[2] = workspace_bytes;
        bytes[3] = 2 * io_bytes * batch;
    }

    void startup_memory(uint64_t bytes[4]) const {
        std::copy(startup_used.begin(), startup_used.end(), bytes);
    }

    void initialize(const fs::path& directory, int size, uint32_t flags, const fs::path& cache_dir) {
        if (size != 1 && size != 16) throw std::runtime_error("Supported batches are 1 and 16");
        if (flags & ~(GT_CREATE_GRAPH | GT_CREATE_FP16 | GT_CREATE_FUSE_SILU))
            throw std::runtime_error("Unsupported native runner flags");
        auto stage_start = Clock::now();
        batch = size;
        fp16 = (flags & GT_CREATE_FP16) != 0;
        fuse_silu = (flags & GT_CREATE_FUSE_SILU) != 0;
        io_bytes = rgb_elements * (fp16 ? sizeof(uint16_t) : sizeof(float));
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, 0));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        ContextScope scope(context);
        record_memory(0);
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        auto module_path = directory / "kernels.cubin";
        auto module_size = fs::file_size(module_path);
        if (!module_size || module_size > (16u << 20)) throw std::runtime_error("Invalid CUDA module size");
        std::vector<char> module_data(static_cast<size_t>(module_size));
        std::ifstream kernel_file(module_path, std::ios::binary);
        read(kernel_file, module_data.data(), module_data.size());
        CHECK(cuModuleLoadData(&module, module_data.data()));
        CHECK(cudnnCreate(&cudnn));
        CHECK(cublasLtCreate(&blas));
        CHECK(cudnnSetStream(cudnn, stream));
        load(directory, !cache_dir.empty(), module_data);
        record_memory(1);
        CHECK(cuMemAllocHost(&host_input, io_bytes * batch));
        CHECK(cuMemAllocHost(&host_output, io_bytes * batch));
        setup_ms[0] = milliseconds_since(stage_start);
        stage_start = Clock::now();
        if (plan_cache.status != GT_CACHE_UNAVAILABLE) {
            try {
                plan_cache.open(cache_dir, plan_key);
            } catch (const std::exception& error) {
                plan_cache.unavailable(error);
            }
        }
        prepare();
        if (fuse_silu) prepare_silu();
        CHECK(cuStreamSynchronize(stream));
        record_memory(2);
        setup_ms[1] = milliseconds_since(stage_start);
        stage_start = Clock::now();
        if (scratch) {
            CHECK(cuMemFree(scratch));
            scratch = 0;
        }
        CHECK(cuMemAlloc(&scratch, workspace_bytes));
        if (silu_plan) bind_silu();
        CHECK(cuMemsetD8Async(input, 0, io_bytes * batch, stream));
        forward(stream);
        CHECK(cuStreamSynchronize(stream));
        setup_ms[2] = milliseconds_since(stage_start);
        stage_start = Clock::now();
        if (flags & GT_CREATE_GRAPH) {
            try {
                capture_graph();
            } catch (const std::exception& error) {
                if (!fused_silu_count) throw;
                if (executable) CHECK(cuGraphExecDestroy(executable));
                if (graph) CHECK(cuGraphDestroy(graph));
                executable = nullptr;
                graph = nullptr;
                disable_fusion(error.what());
                capture_graph();
            }
        }
        setup_ms[3] = milliseconds_since(stage_start);
        record_memory(3);
        plan_cache.save(plan_key);
    }

    ~GtRunner() {
        if (!context) return;
        // Destruction also works if initialization failed partway through.
        if (cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
        if (stream) cuStreamSynchronize(stream);
        if (executable) cuGraphExecDestroy(executable);
        if (graph) cuGraphDestroy(graph);
        silu_bindings.clear();
        silu_plan.reset();
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
    std::map<ShapeKey, GemmChoice> gemm_algorithms;
    std::map<ShapeKey, std::unique_ptr<Convolution>> convolutions;
    std::map<std::string, CUfunction> kernels;
    std::vector<CUfunction> kernel_by_op;
    CUfunction add_bias = nullptr;
    CUfunction mean_project = nullptr;
    size_t arena_bytes = 0;
    size_t weight_bytes = 0;
    size_t workspace_bytes = 0;
    std::array<uint64_t, 4> startup_used{};
    gt::Digest plan_key{};
    std::unique_ptr<gt::SiluPlan> silu_plan;
    std::vector<size_t> silu_sites;
    std::vector<std::unique_ptr<gt::BackendDescriptor>> silu_bindings;

    void ensure_tuning_workspace() {
        if (scratch) return;
        CHECK(cuMemAlloc(&scratch, scratch_bytes));
        CHECK(cuMemsetD8Async(arena, 0x28, arena_bytes, stream));
        record_memory(1);
    }

    void record_memory(size_t stage) {
        size_t available, total;
        CHECK(cuMemGetInfo(&available, &total));
        startup_used[stage] = total - available;
    }

    void capture_graph() {
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

    void load(const fs::path& directory, bool cache_enabled, const std::vector<char>& module_data) {
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
        if (cache_enabled) {
            try {
                gt::Sha256 hash;
                const char build_id[] = __DATE__ " " __TIME__;
                hash.add(build_id);
                hash.add(module_data.data(), module_data.size());
                hash.add(header);
                for (const auto& t : tensors) {
                    // All arena placements have the same alignment. Plan geometry
                    // depends on tensor shapes and strides, independently of offsets.
                    hash.add(std::array<int, 3>{t.h, t.w, t.c});
                }
                hash.add(ops.data(), ops.size() * sizeof(Op));
                hash.add(staging.data(), staging.size());
                hash.add(batch);
                hash.add(fp16);
                hash.add(scratch_bytes);
                CUuuid uuid{};
                CHECK(cuDeviceGetUuid(&uuid, device));
                hash.add(uuid);
                hash.add(gt::driver_file_version());
                hash.add(cudnnGetVersion());
                hash.add(cublasLtGetVersion());
                plan_key = hash.finish();
            } catch (const std::exception& error) {
                plan_cache.unavailable(error);
            }
        }
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
        auto key = ShapeKey{cin, out.c, count, a.c, op.weight[1] >= 0};
        if (auto cached = gemm_algorithms.find(key); cached != gemm_algorithms.end()) {
            plan->algorithm = cached->second.algorithm;
            plan->workspace = cached->second.workspace;
            matmuls[index] = std::move(plan);
            return;
        }
        if (auto saved = plan_cache.gemms.find(key); saved != plan_cache.gemms.end()) {
            const auto& choice = saved->second;
            cublasLtMatmulHeuristicResult_t checked{};
            if (choice.workspace <= scratch_bytes
                && cublasLtMatmulAlgoCheck(blas, plan->desc, plan->kernel, plan->input, plan->output,
                                           plan->output, &choice.algorithm, &checked) == CUBLAS_STATUS_SUCCESS
                && checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize == choice.workspace) {
                plan->algorithm = choice.algorithm;
                plan->workspace = choice.workspace;
                workspace_bytes = std::max(workspace_bytes, plan->workspace);
                gemm_algorithms.emplace(key, choice);
                matmuls[index] = std::move(plan);
                return;
            }
        }
        ensure_tuning_workspace();
        plan_cache.record_tuning();
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
        auto selected = GemmChoice{plan->algorithm, plan->workspace};
        gemm_algorithms.emplace(key, selected);
        plan_cache.gemms.insert_or_assign(key, selected);
        matmuls[index] = std::move(plan);
    }

    cudnnStatus_t convolution(const Convolution& p, const Op& op) {
        return cudnnConvolutionForward(cudnn, &one, p.input, pointer(tensor(op.input[0])),
            p.weight, pointer(weight(op.weight[0])), p.conv, p.algorithm, pointer(scratch),
            p.workspace, &zero, p.output, pointer(tensor(op.out)));
    }

    void prepare_conv(size_t index, const Tensor& in, const Tensor& out, const Op& op, TuningTimer& timer) {
        auto key = ShapeKey{in.h, in.w, in.c, out.c, op.x};
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
            if (auto saved = plan_cache.convolutions.find(key); saved != plan_cache.convolutions.end()) {
                const auto& choice = saved->second;
                size_t required = 0;
                auto algorithm = static_cast<cudnnConvolutionFwdAlgo_t>(choice.algorithm);
                auto math = static_cast<cudnnMathType_t>(choice.math);
                if (choice.algorithm < CUDNN_CONVOLUTION_FWD_ALGO_COUNT
                    && choice.math <= CUDNN_FMA_MATH && choice.workspace <= scratch_bytes
                    && cudnnSetConvolutionMathType(p.conv, math) == CUDNN_STATUS_SUCCESS
                    && cudnnGetConvolutionForwardWorkspaceSize(cudnn, p.input, p.weight, p.conv,
                                                               p.output, algorithm, &required) == CUDNN_STATUS_SUCCESS
                    && required == choice.workspace) {
                    p.algorithm = algorithm;
                    p.workspace = required;
                    workspace_bytes = std::max(workspace_bytes, p.workspace);
                    conv_by_op[index] = &p;
                    return;
                }
            }
            ensure_tuning_workspace();
            plan_cache.record_tuning();
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
            plan_cache.convolutions.insert_or_assign(
                key, gt::ConvChoice{uint32_t(fastest->algo), uint32_t(fastest->mathType), p.workspace});
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

    void disable_fusion(std::string diagnostic) {
        silu_bindings.clear();
        silu_plan.reset();
        fused_silu_count = 0;
        fusion_diagnostic = std::move(diagnostic);
    }

    void prepare_silu() {
        std::vector<int> uses(tensors.size());
        for (const auto& op : ops)
            for (int id : op.input)
                if (id >= 0) ++uses[id];
        for (size_t i = 0; i + 1 < ops.size(); ++i) {
            const auto& conv = ops[i];
            const auto& silu = ops[i + 1];
            if (conv.kind != Conv || conv.x != 3 || conv.weight[1] >= 0 || silu.kind != Silu
                || silu.input[0] != conv.out || uses[conv.out] != 1) continue;
            const auto& x = tensors[conv.input[0]];
            const auto& y = tensors[silu.out];
            if (x.h != tile || x.w != tile || x.c != 32 || y.h != tile || y.w != tile || y.c != 32)
                continue;
            const auto x_end = x.offset + x.elements() * 2;
            const auto y_end = y.offset + y.elements() * 2;
            if (std::max(x.offset, y.offset) < std::min(x_end, y_end)) continue;
            silu_sites.push_back(i);
        }
        if (silu_sites.empty()) {
            fusion_diagnostic = "Schedule has no compatible non-aliasing convolution/SiLU sites";
            return;
        }
        try {
            auto candidate = std::make_unique<gt::SiluPlan>();
            candidate->initialize(cudnn, batch, fp16, scratch_bytes);
            workspace_bytes = std::max(workspace_bytes, candidate->workspace);
            silu_plan = std::move(candidate);
        } catch (const std::exception& error) {
            disable_fusion(error.what());
        }
    }

    void bind_silu() {
        try {
            silu_bindings.resize(ops.size());
            for (auto i : silu_sites) {
                const auto& conv = ops[i];
                silu_bindings[i] = silu_plan->bind(pointer(tensor(conv.input[0])),
                    pointer(weight(conv.weight[0])), pointer(tensor(ops[i + 1].out)), pointer(scratch));
            }
            silu_plan->run(cudnn, *silu_bindings[silu_sites.front()]);
            CHECK(cuStreamSynchronize(stream));
            fused_silu_count = static_cast<int>(silu_sites.size());
        } catch (const std::exception& error) {
            disable_fusion(error.what());
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
            if (fused_silu_count) {
                if (silu_bindings[i]) {
                    silu_plan->run(cudnn, *silu_bindings[i]);
                    continue;
                }
                if (i && silu_bindings[i - 1]) continue;
            }
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
    return gt_create_ex(directory, batch, graph ? GT_CREATE_GRAPH : 0, nullptr);
}

GtRunner* gt_create_fp16(const char* directory, int batch, int graph) {
    return gt_create_ex(directory, batch, GT_CREATE_FP16 | (graph ? GT_CREATE_GRAPH : 0), nullptr);
}

GtRunner* gt_create_ex(const char* directory, int batch, uint32_t flags, const char* cache_dir) {
    GtRunner* result = nullptr;
    report_errors([&] {
        auto runner = std::make_unique<GtRunner>();
        runner->initialize(fs::u8path(directory), batch, flags,
                           cache_dir ? fs::u8path(cache_dir) : fs::path{});
        result = runner.release();
    });
    return result;
}

GtCacheStatus gt_cache_status(const GtRunner* runner) { return runner->plan_cache.status; }
const char* gt_cache_message(const GtRunner* runner) { return runner->plan_cache.diagnostic.c_str(); }
void gt_setup_timings(const GtRunner* runner, double milliseconds[4]) {
    std::copy(runner->setup_ms.begin(), runner->setup_ms.end(), milliseconds);
}
int gt_fused_silu_count(const GtRunner* runner) { return runner->fused_silu_count; }
const char* gt_fusion_message(const GtRunner* runner) { return runner->fusion_diagnostic.c_str(); }

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
    if (argc < 5) {
        std::cerr << "Usage: gaterv3 <bundle-directory> <batch: 1|16> <input> <output> "
                     "[--fp16] [--cache-dir <directory>] [--fuse-silu]\n";
        return 2;
    }
    return report_errors([&] {
        uint32_t flags = GT_CREATE_GRAPH;
        const char* cache_dir = nullptr;
        for (int i = 5; i < argc; ++i) {
            std::string option = argv[i];
            if (option == "--fp16") flags |= GT_CREATE_FP16;
            else if (option == "--fuse-silu") flags |= GT_CREATE_FUSE_SILU;
            else if (option == "--cache-dir" && i + 1 < argc) cache_dir = argv[++i];
            else throw std::runtime_error("Invalid option " + option);
        }
        std::unique_ptr<GtRunner> runner(gt_create_ex(argv[1], std::stoi(argv[2]), flags, cache_dir));
        if (!runner) throw std::runtime_error(error_text);
        if (*gt_cache_message(runner.get()))
            std::cerr << "Plan cache: " << gt_cache_message(runner.get()) << '\n';
        if (flags & GT_CREATE_FUSE_SILU) {
            std::cerr << "Convolution/SiLU sites: " << gt_fused_silu_count(runner.get()) << '\n';
            if (*gt_fusion_message(runner.get()))
                std::cerr << gt_fusion_message(runner.get()) << '\n';
        }
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
