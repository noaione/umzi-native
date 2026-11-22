#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cuda.h>
#include <cudnn.h>

#include "gfisrv2_images.h"
#include "cufft_api.h"
#include "../../common/cuda_module.h"
#include "../../runtime/binary.h"
#include "../../runtime/kernel.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <iterator>
#include <map>
#include <memory>
#include <string_view>

namespace {
constexpr int channels = 48, blocks = 24, scale = 4, hidden = 128, fourier = 30, band = 6, threads = 256;
constexpr std::array<uint32_t, 7> configuration{3, 3, scale, channels, blocks, hidden, fourier};
constexpr int r2c = 0x2a, c2r = 0x2c;
using Clock = std::chrono::steady_clock;

void check(CUresult status, const char* operation) {
    if (status == CUDA_ERROR_OUT_OF_MEMORY)
        throw umzi::AllocationError(std::string(operation) + ": out of memory; try a smaller tile or batch");
    umzi::check_cuda(status, operation);
}
void check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
}
void fft_check(int status, const char* operation) {
    if (status == 2)
        throw umzi::AllocationError(std::string(operation) + ": cuFFT allocation failed; try a smaller tile or batch");
    if (status) throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(status));
}
#define CHECK(expr) check((expr), #expr)
#define FFT_CHECK(expr) fft_check((expr), #expr)

struct ContextScope {
    explicit ContextScope(CUcontext context) { CHECK(cuCtxPushCurrent(context)); }
    ~ContextScope() {
        CUcontext previous;
        cuCtxPopCurrent(&previous);
    }
};

struct Shape {
    int n = 0, h = 0, w = 0;
    bool operator==(Shape other) const { return n == other.n && h == other.h && w == other.w; }
    int pixels() const { return h * w; }
    uint64_t rgb_bytes() const { return uint64_t(n) * pixels() * 3 * sizeof(float); }
    UmziShape nchw() const { return {4, {uint32_t(n), 3, uint32_t(h), uint32_t(w)}}; }
};

Shape checked(const UmziShape& shape) {
    const auto n = shape.dimensions[0], h = shape.dimensions[2], w = shape.dimensions[3];
    if (shape.rank != 4 || shape.dimensions[1] != 3 || !n || !h || !w || n > 4096 || h > 65536 || w > 65536)
        throw std::runtime_error("GFISRv2 expects positive RGB NCHW dimensions within the native indexing limits");
    if (uint64_t(n) * h * w * 2 * hidden > INT_MAX - threads)
        throw std::runtime_error("Requested GFISRv2 shape exceeds native tensor indexing limits");
    return {int(n), int(h), int(w)};
}

struct Norm { size_t scale, offset; };
struct Filter { size_t weight, bias; };
struct Fourier { Norm pre; Filter position, project; Norm post; };
struct Block { Norm norm; Filter expand; Fourier fft; std::array<Filter, 3> bands; Filter project; size_t gamma; };

// Section 3 stores FP32 OIHW weights and affine parameters at 256-byte offsets.
struct Network {
    size_t bytes = 0;
    Filter first, tail_expand, tail_project, last;
    std::array<Block, blocks> body{};

    Network() {
        first = conv(3, channels);
        for (auto& block : body) {
            block.norm = norm(channels);
            block.expand = conv(channels, 2 * hidden);
            block.fft = {norm(2 * fourier), conv(1, 2 * fourier), conv(2 * fourier, 2 * fourier, 1), norm(fourier)};
            block.bands = {conv(1, band), conv(1, band, 11), conv(1, band, 11)};
            block.project = conv(hidden, channels);
            block.gamma = take(channels);
        }
        tail_expand = conv(channels, 2 * channels);
        tail_project = conv(2 * channels, channels);
        last = conv(channels, 3 * scale * scale);
    }

    size_t take(size_t count) {
        const size_t offset = (bytes + 255) & ~size_t(255);
        bytes = offset + count * sizeof(float);
        return offset;
    }
    Norm norm(int c) { return {take(c), take(c)}; }
    Filter conv(int cin, int cout, int taps = 9) { return {take(size_t(cin) * cout * taps), take(cout)}; }
};

struct Convolution {
    cudnnTensorDescriptor_t input = nullptr, output = nullptr;
    cudnnFilterDescriptor_t filter = nullptr;
    cudnnConvolutionDescriptor_t convolution = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm{};
    size_t workspace = 0;
    ~Convolution() {
        if (convolution) cudnnDestroyConvolutionDescriptor(convolution);
        if (filter) cudnnDestroyFilterDescriptor(filter);
        if (output) cudnnDestroyTensorDescriptor(output);
        if (input) cudnnDestroyTensorDescriptor(input);
    }
};

struct FftPlan {
    int handle = 0;
    bool created = false;
    CUdeviceptr workspace = 0;
    size_t bytes = 0;

    void setup(const gfi::CufftApi& api, Shape shape, int transform) {
        FFT_CHECK(api.create(&handle));
        created = true;
        FFT_CHECK(api.auto_allocation(handle, 0));
        int sizes[] = {shape.h, shape.w};
        FFT_CHECK(api.make_plan(handle, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, transform, shape.n * fourier, &bytes));
        if (bytes) {
            CHECK(cuMemAlloc(&workspace, bytes));
            FFT_CHECK(api.work_area(handle, reinterpret_cast<void*>(workspace)));
        }
    }

    void release(const gfi::CufftApi& api) {
        if (created) api.destroy(handle);
        if (workspace) cuMemFree(workspace);
        *this = {};
    }
};

constexpr const char* kernel_names[] = {
    "rms_norm", "depthwise", "copy_channels", "pack_fft", "unpack_fft", "scale_fft", "add_bias",
    "gate", "residual", "apply_silu", "add_features", "shuffle_rgb",
};
enum Kernel { Rms, Depthwise, CopyChannels, Pack, Unpack, ScaleFft, Bias, Gate, Residual, Silu, Add, Shuffle, KernelCount };
static_assert(std::size(kernel_names) == KernelCount);

class Session final : public umzi::KernelSession {
    Network model;
    bool graph_enabled = false, pending = false, unexact_erf = false;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUstream stream = nullptr;
    CUevent complete = nullptr;
    cudnnHandle_t cudnn = nullptr;
    std::unique_ptr<gfi::CufftApi> fft;
    int fft_version = 0;
    std::array<CUfunction, KernelCount> functions{};
    CUdeviceptr weights = 0;

    Shape shape{};
    int bins = 0;
    float fft_scale = 0;
    CUdeviceptr stage_in = 0, stage_out = 0, cudnn_scratch = 0;
    CUdeviceptr initial = 0, state = 0, norm = 0, projected = 0, gated = 0;
    CUdeviceptr spectrum = 0, packed = 0, spectral_out = 0;
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    FftPlan forward_fft, inverse_fft;
    using Key = std::array<int, 5>;
    std::map<Key, std::unique_ptr<Convolution>> plans;
    size_t activation_bytes = 0, cudnn_workspace = 0;
    std::array<double, 4> setup_ms{};

public:
    void open(const umzi::ModelPackage& package, const UmziSessionOptions& options) {
        if (package.schema != 1) throw std::runtime_error("Unsupported GFISRv2 model payload version");
        umzi::BinaryReader architecture(package.section(1));
        for (auto value : configuration)
            if (architecture.number<uint32_t>() != value) throw std::runtime_error("Unsupported GFISRv2 model architecture");
        if (architecture.size() || package.section(4).size() != 32)
            throw std::runtime_error("Invalid GFISRv2 metadata");
        const auto data = package.section(3);
        if (!package.section(2).empty() || data.size() != model.bytes)
            throw std::runtime_error("Invalid GFISRv2 weight layout");
        if (options.precision != UMZI_FP32 && options.precision != UMZI_MIXED)
            throw std::runtime_error("GFISRv2 currently requires FP32 precision");
        for (uint32_t i = 0; i < options.kernel_option_count; ++i) {
            const auto& option = options.kernel_options[i];
            const std::string_view name = option.name ? option.name : "";
            if (name != "unexact_erf")
                throw std::runtime_error("Unknown GFISRv2 option: " + std::string(name));
            if (option.value != 0 && option.value != 1)
                throw std::runtime_error("GFISRv2 unexact_erf must be 0 or 1");
            unexact_erf = option.value == 1;
        }
        graph_enabled = (options.flags & UMZI_CUDA_GRAPH) != 0;
        fft = std::make_unique<gfi::CufftApi>();
        FFT_CHECK(fft->version(&fft_version));
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        ContextScope scope(context);
        umzi::load_kernel_module(device, module, gfisrv2_images::embedded_cubins, gfisrv2_images::embedded_ptx);
        for (int i = 0; i < KernelCount; ++i) {
            const char* name = i == Unpack && unexact_erf ? "unpack_fft_unexact" : kernel_names[i];
            CHECK(cuModuleGetFunction(&functions[i], module, name));
        }
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        CHECK(cudnnCreate(&cudnn));
        CHECK(cuMemAlloc(&weights, data.size()));
        CHECK(cuMemcpyHtoD(weights, data.data(), data.size()));
    }

    ~Session() override {
        if (!context) return;
        if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            if (pending) cuEventSynchronize(complete);
            if (stream) cuStreamSynchronize(stream);
            release_shape();
            if (cudnn) cudnnDestroy(cudnn);
            if (complete) cuEventDestroy(complete);
            if (stream) cuStreamDestroy(stream);
            if (weights) cuMemFree(weights);
            if (module) cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        cuDevicePrimaryCtxRelease(device);
    }

private:
    void release_shape() {
        if (executable) cuGraphExecDestroy(executable);
        if (graph) cuGraphDestroy(graph);
        executable = nullptr;
        graph = nullptr;
        forward_fft.release(*fft);
        inverse_fft.release(*fft);
        plans.clear();
        for (auto* buffer : {&stage_in, &stage_out, &cudnn_scratch, &initial, &state, &norm, &projected,
                             &gated, &spectrum, &packed, &spectral_out}) {
            if (*buffer) cuMemFree(*buffer);
            *buffer = 0;
        }
        shape = {};
        activation_bytes = cudnn_workspace = 0;
        setup_ms = {};
    }

    void wait() {
        if (!pending) return;
        CHECK(cuEventSynchronize(complete));
        pending = false;
    }

    void activation(CUdeviceptr& buffer, size_t count) {
        const auto bytes = count * sizeof(float);
        CHECK(cuMemAlloc(&buffer, bytes));
        activation_bytes += bytes;
    }

    void make_conv(int h, int w, int cin, int cout, int k) {
        auto plan = std::make_unique<Convolution>();
        auto& p = *plan;
        CHECK(cudnnCreateTensorDescriptor(&p.input));
        CHECK(cudnnCreateTensorDescriptor(&p.output));
        CHECK(cudnnCreateFilterDescriptor(&p.filter));
        CHECK(cudnnCreateConvolutionDescriptor(&p.convolution));
        CHECK(cudnnSetTensor4dDescriptor(p.input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape.n, cin, h, w));
        CHECK(cudnnSetTensor4dDescriptor(p.output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape.n, cout, h, w));
        CHECK(cudnnSetFilter4dDescriptor(p.filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, cout, cin, k, k));
        CHECK(cudnnSetConvolution2dDescriptor(p.convolution, k / 2, k / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
        CHECK(cudnnSetConvolutionMathType(p.convolution, CUDNN_FMA_MATH));
        // The reference uses implicit GEMM for RGB and spectral projection,
        // and Winograd for the wider spatial convolutions on the target GPU.
        const auto preferred = k == 1 || cin == 3 ? CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
                                                  : CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        const auto status = cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, p.input, p.filter, p.convolution, p.output, preferred, &p.workspace);
        p.algorithm = preferred;
        if (status != CUDNN_STATUS_SUCCESS) {
            if (status != CUDNN_STATUS_NOT_SUPPORTED) CHECK(status);
            p.algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                cudnn, p.input, p.filter, p.convolution, p.output, p.algorithm, &p.workspace));
        }
        cudnn_workspace = std::max(cudnn_workspace, p.workspace);
        plans.emplace(Key{h, w, cin, cout, k}, std::move(plan));
    }

    static double elapsed(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    void bind_stream(CUstream target) {
        CHECK(cudnnSetStream(cudnn, target));
        FFT_CHECK(fft->set_stream(forward_fft.handle, reinterpret_cast<void*>(target)));
        FFT_CHECK(fft->set_stream(inverse_fft.handle, reinterpret_cast<void*>(target)));
    }

    void prepare(Shape requested) {
        wait();
        if (shape == requested) return;
        release_shape();
        try {
            auto start = Clock::now();
            shape = requested;
            const size_t pixels = size_t(shape.n) * shape.pixels();
            bins = shape.h * (shape.w / 2 + 1);
            fft_scale = float(1.0 / std::sqrt(double(shape.pixels())));
            activation(initial, pixels * channels);
            activation(state, pixels * channels);
            activation(norm, pixels * channels);
            activation(projected, pixels * 2 * hidden);
            activation(gated, pixels * hidden);
            const size_t frequency_size = size_t(shape.n) * bins * 2 * fourier;
            activation(spectrum, frequency_size);
            activation(packed, frequency_size);
            activation(spectral_out, frequency_size);
            CHECK(cuMemAlloc(&stage_in, shape.rgb_bytes()));
            CHECK(cuMemAlloc(&stage_out, shape.rgb_bytes() * scale * scale));
            setup_ms[0] = elapsed(start);

            start = Clock::now();
            const int h = shape.h, w = shape.w;
            make_conv(h, w, 3, channels, 3);
            make_conv(h, w, channels, 2 * hidden, 3);
            make_conv(h, w, hidden, channels, 3);
            make_conv(h, w, channels, 2 * channels, 3);
            make_conv(h, w, 2 * channels, channels, 3);
            make_conv(h, w, channels, 3 * scale * scale, 3);
            make_conv(h, w / 2 + 1, 2 * fourier, 2 * fourier, 1);
            if (cudnn_workspace) CHECK(cuMemAlloc(&cudnn_scratch, cudnn_workspace));
            forward_fft.setup(*fft, shape, r2c);
            inverse_fft.setup(*fft, shape, c2r);
            bind_stream(stream);
            setup_ms[1] = elapsed(start);

            start = Clock::now();
            CHECK(cuMemsetD8Async(stage_in, 0, shape.rgb_bytes(), stream));
            forward(stage_in, stage_out, stream);
            CHECK(cuStreamSynchronize(stream));
            setup_ms[2] = elapsed(start);

            if (graph_enabled) {
                start = Clock::now();
                CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
                try {
                    forward(stage_in, stage_out, stream);
                } catch (...) {
                    cuStreamEndCapture(stream, &graph);
                    throw;
                }
                CHECK(cuStreamEndCapture(stream, &graph));
                CHECK(cuGraphInstantiate(&executable, graph, 0));
                CHECK(cuGraphUpload(executable, stream));
                CHECK(cuStreamSynchronize(stream));
                setup_ms[3] = elapsed(start);
            }
        } catch (...) {
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        }
    }

    template<class... Args>
    void launch(CUstream target, Kernel id, int count, Args... args) {
        void* params[] = {&args..., &count};
        CHECK(cuLaunchKernel(functions[id], (count + threads - 1) / threads, 1, 1, threads, 1, 1, 0,
                             target, params, nullptr));
    }

    CUdeviceptr channel_offset(CUdeviceptr base, int channel) const {
        return base + uint64_t(channel) * shape.pixels() * sizeof(float);
    }

    void conv(CUstream target, CUdeviceptr out, CUdeviceptr in, int h, int w, int cin, int cout, int k, Filter filter) {
        const auto& p = *plans.at(Key{h, w, cin, cout, k});
        const float one = 1.f, zero = 0.f;
        CHECK(cudnnConvolutionForward(cudnn, &one, p.input, reinterpret_cast<void*>(in), p.filter,
            reinterpret_cast<void*>(weights + filter.weight), p.convolution, p.algorithm,
            reinterpret_cast<void*>(cudnn_scratch), p.workspace, &zero, p.output, reinterpret_cast<void*>(out)));
        launch(target, Bias, shape.n * h * w * cout, out, weights + filter.bias, cout, h * w);
    }

    void rms(CUstream target, CUdeviceptr out, CUdeviceptr in, Norm params, int c, int pixels, int out_stride) {
        launch(target, Rms, shape.n * pixels, out, in, weights + params.scale, weights + params.offset,
               c, c, out_stride, pixels, float(1.0 / std::sqrt(double(c))));
    }

    void fourier_unit(CUstream target, const Fourier& params, CUdeviceptr in, CUdeviceptr out) {
        const int pixels = shape.pixels(), frequency = shape.n * fourier * bins;
        launch(target, CopyChannels, shape.n * fourier * pixels, gated, in, fourier, 2 * hidden, pixels);
        FFT_CHECK(fft->r2c(forward_fft.handle, reinterpret_cast<void*>(gated), reinterpret_cast<void*>(spectrum)));
        launch(target, Pack, frequency, packed, spectrum, fourier, bins, fft_scale);
        rms(target, spectral_out, packed, params.pre, 2 * fourier, bins, 2 * fourier);
        launch(target, Depthwise, frequency * 2, packed, spectral_out, weights + params.position.weight,
               weights + params.position.bias, 2 * fourier, 2 * fourier, 2 * fourier, shape.h, shape.w / 2 + 1,
               3, 3, 1);
        conv(target, spectral_out, packed, shape.h, shape.w / 2 + 1, 2 * fourier, 2 * fourier, 1, params.project);
        launch(target, Unpack, frequency, spectrum, spectral_out, bins);
        FFT_CHECK(fft->c2r(inverse_fft.handle, reinterpret_cast<void*>(spectrum), reinterpret_cast<void*>(gated)));
        launch(target, ScaleFft, shape.n * fourier * pixels, gated, fft_scale);
        rms(target, out, gated, params.post, fourier, pixels, channels);
    }

    void forward(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const int pixels = shape.pixels(), area = shape.n * pixels, h = shape.h, w = shape.w;
        conv(target, initial, input, h, w, 3, channels, 3, model.first);
        CHECK(cuMemcpyDtoDAsync(state, initial, uint64_t(area) * channels * sizeof(float), target));
        for (int i = 0; i < blocks; ++i) {
            const auto& block = model.body[i];
            rms(target, norm, state, block.norm, channels, pixels, channels);
            conv(target, projected, norm, h, w, channels, 2 * hidden, 3, block.expand);
            int offset = 0;
            for (int slot = 0; slot < 4; ++slot) {
                const int kind = (slot + i) % 4;
                const auto in = channel_offset(projected, 2 * hidden - channels + offset);
                const auto out = channel_offset(norm, offset);
                if (!kind) {
                    fourier_unit(target, block.fft, in, out);
                    offset += fourier;
                } else {
                    const auto& filter = block.bands[kind - 1];
                    const int kh = kind == 1 ? 3 : kind == 2 ? 1 : 11;
                    const int kw = kind == 1 ? 3 : kind == 2 ? 11 : 1;
                    launch(target, Depthwise, area * band, out, in, weights + filter.weight, weights + filter.bias,
                           band, 2 * hidden, channels, h, w, kh, kw, 0);
                    offset += band;
                }
            }
            launch(target, Gate, area * hidden, gated, projected, norm, hidden, channels, pixels);
            conv(target, projected, gated, h, w, hidden, channels, 3, block.project);
            launch(target, Residual, area * channels, state, projected, weights + block.gamma, channels, pixels);
        }
        conv(target, projected, state, h, w, channels, 2 * channels, 3, model.tail_expand);
        launch(target, Silu, area * 2 * channels, projected);
        conv(target, norm, projected, h, w, 2 * channels, channels, 3, model.tail_project);
        launch(target, Add, area * channels, state, norm, initial);
        conv(target, projected, state, h, w, channels, 3 * scale * scale, 3, model.last);
        launch(target, Shuffle, area * 3 * scale * scale, output, projected, h, w);
    }

    void execute(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        if (!executable) {
            bind_stream(target);
            forward(input, output, target);
            return;
        }
        const auto bytes = shape.rgb_bytes();
        if (input != stage_in) CHECK(cuMemcpyDtoDAsync(stage_in, input, bytes, target));
        CHECK(cuGraphLaunch(executable, target));
        if (output != stage_out) CHECK(cuMemcpyDtoDAsync(output, stage_out, bytes * scale * scale, target));
    }

public:
    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "gfisrv2", scale, 0}; }
    UmziIo tensors() const override {
        return {sizeof(UmziIo), {"input", {4, {0, 3, 0, 0}}, UMZI_FLOAT32, 0},
                                {"output", {4, {0, 3, 0, 0}}, UMZI_FLOAT32, 0}};
    }

    UmziIo describe(const UmziShape& request) const override {
        const auto concrete = checked(request);
        auto io = tensors();
        io.input.shape = request;
        io.input.bytes = concrete.rgb_bytes();
        io.output.shape = Shape{concrete.n, scale * concrete.h, scale * concrete.w}.nchw();
        io.output.bytes = io.input.bytes * scale * scale;
        return io;
    }

    void prepare_shape(const UmziShape& request) override {
        ContextScope scope(context);
        prepare(checked(request));
    }

    void run_host(const UmziShape& request, const void* input, void* output) override {
        ContextScope scope(context);
        prepare(checked(request));
        const auto bytes = shape.rgb_bytes();
        try {
            CHECK(cuMemcpyHtoDAsync(stage_in, input, bytes, stream));
            execute(stage_in, stage_out, stream);
            CHECK(cuMemcpyDtoHAsync(output, stage_out, bytes * scale * scale, stream));
            CHECK(cuStreamSynchronize(stream));
        } catch (...) {
            cuStreamSynchronize(stream);
            throw;
        }
    }

    void enqueue_device(const UmziShape& request, uint64_t input, uint64_t output, UmziStream target) override {
        ContextScope scope(context);
        prepare(checked(request));
        try {
            execute(input, output, target);
            CHECK(cuEventRecord(complete, target));
            pending = true;
        } catch (...) {
            cuStreamSynchronize(target);
            throw;
        }
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo info{};
        info.struct_size = sizeof(info);
        info.cache_status = UMZI_CACHE_DISABLED;
        info.buffer_bytes[0] = model.bytes;
        if (shape.n) {
            info.prepared_input = shape.nchw();
            info.buffer_bytes[1] = activation_bytes;
            info.buffer_bytes[2] = cudnn_workspace + forward_fft.bytes + inverse_fft.bytes;
            info.buffer_bytes[3] = shape.rgb_bytes() * (1 + scale * scale);
            std::copy(setup_ms.begin(), setup_ms.end(), info.shape_setup_ms);
        }
        return info;
    }

    uint32_t diagnostic_count() const override { return 3; }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "cufft_version", fft_version};
        if (index == 1)
            return {sizeof(UmziDiagnostic), "cufft_workspace_bytes", int64_t(forward_fft.bytes + inverse_fft.bytes)};
        if (index == 2) return {sizeof(UmziDiagnostic), "unexact_erf", unexact_erf};
        throw std::runtime_error("Invalid GFISRv2 diagnostic index");
    }
    const char* kernel_message() const override { return "GFISRv2 FP32 with cuFFT"; }
};
} // namespace

std::unique_ptr<umzi::KernelSession> umzi::make_gfisrv2(const ModelPackage& package, const UmziSessionOptions& options) {
    auto session = std::make_unique<Session>();
    session->open(package, options);
    return session;
}
