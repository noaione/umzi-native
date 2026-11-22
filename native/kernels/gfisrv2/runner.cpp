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
    uint64_t rgb_bytes(size_t element) const { return uint64_t(n) * pixels() * 3 * element; }
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
    "rms_norm_half", "rms_norm_to_half", "depthwise_half", "copy_channels_to_float", "pack_fft_half",
    "unpack_fft_half", "scale_fft_to_half", "add_bias_half", "gate_half", "gate_mixed", "residual_half",
    "residual_mixed", "apply_silu_half", "add_features_half", "add_features_mixed", "cast_to_half",
    "cast_to_float", "add_half_residual", "shuffle_rgb_half", "shuffle_rgb_mixed",
};
enum Kernel {
    Rms, Depthwise, CopyChannels, Pack, Unpack, ScaleFft, Bias, Gate, Residual, Silu, Add, Shuffle,
    RmsHalf, RmsToHalf, DepthwiseHalf, CopyChannelsToFloat, PackHalf, UnpackHalf, ScaleFftToHalf,
    BiasHalf, GateHalf, GateMixed, ResidualHalf, ResidualMixed, SiluHalf, AddHalf, AddMixed,
    ToHalf, ToFloat, AddHalfResidual, ShuffleHalf, ShuffleMixed, KernelCount
};
static_assert(std::size(kernel_names) == KernelCount);

class Session final : public umzi::KernelSession {
    Network model;
    bool graph_enabled = false, pending = false, unexact_erf = false;
    uint32_t precision = UMZI_FP32;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUstream stream = nullptr;
    CUevent complete = nullptr;
    cudnnHandle_t cudnn = nullptr;
    std::unique_ptr<gfi::CufftApi> fft;
    int fft_version = 0;
    std::array<CUfunction, KernelCount> functions{};
    CUdeviceptr weights = 0, weights_half = 0;

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
        precision = options.precision;
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
            const char* name = kernel_names[i];
            if (unexact_erf && i == Unpack) name = "unpack_fft_unexact";
            if (unexact_erf && i == UnpackHalf) name = "unpack_fft_half_unexact";
            CHECK(cuModuleGetFunction(&functions[i], module, name));
        }
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        CHECK(cudnnCreate(&cudnn));
        CHECK(cuMemAlloc(&weights, data.size()));
        CHECK(cuMemcpyHtoD(weights, data.data(), data.size()));
        if (reduced()) {
            CHECK(cuMemAlloc(&weights_half, data.size() / 2));
            launch(stream, ToHalf, int(data.size() / sizeof(float)), weights_half, weights);
            CHECK(cuStreamSynchronize(stream));
            if (half_io()) {
                CHECK(cuMemFree(weights));
                weights = 0;
            }
        }
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
            if (weights_half) cuMemFree(weights_half);
            if (module) cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        cuDevicePrimaryCtxRelease(device);
    }

private:
    bool half_io() const { return precision == UMZI_FP16; }
    bool mixed() const { return precision == UMZI_MIXED; }
    bool reduced() const { return precision != UMZI_FP32; }
    size_t element(bool half) const { return half ? sizeof(uint16_t) : sizeof(float); }
    CUdeviceptr weight(size_t offset, bool half) const {
        return half ? weights_half + offset / 2 : weights + offset;
    }
    bool grouped_half_spectral_conv() const {
        // Match the eager FP16 depthwise backend for a stride-1 3x3 filter.
        return half_io() && (shape.h == 1 || shape.w / 2 + 1 >= 7);
    }

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

    void activation(CUdeviceptr& buffer, size_t count, bool half = false) {
        const auto bytes = count * element(half);
        CHECK(cuMemAlloc(&buffer, bytes));
        activation_bytes += bytes;
    }

    void make_conv(int h, int w, int cin, int cout, int k, bool half, int groups = 1) {
        auto plan = std::make_unique<Convolution>();
        auto& p = *plan;
        CHECK(cudnnCreateTensorDescriptor(&p.input));
        CHECK(cudnnCreateTensorDescriptor(&p.output));
        CHECK(cudnnCreateFilterDescriptor(&p.filter));
        CHECK(cudnnCreateConvolutionDescriptor(&p.convolution));
        const auto dtype = half ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
        CHECK(cudnnSetTensor4dDescriptor(p.input, CUDNN_TENSOR_NCHW, dtype, shape.n, cin, h, w));
        CHECK(cudnnSetTensor4dDescriptor(p.output, CUDNN_TENSOR_NCHW, dtype, shape.n, cout, h, w));
        CHECK(cudnnSetFilter4dDescriptor(p.filter, dtype, CUDNN_TENSOR_NCHW, cout, cin / groups, k, k));
        CHECK(cudnnSetConvolution2dDescriptor(p.convolution, k / 2, k / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
        if (groups != 1) CHECK(cudnnSetConvolutionGroupCount(p.convolution, groups));
        CHECK(cudnnSetConvolutionMathType(p.convolution, half ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH));
        // The reference uses implicit GEMM for RGB and spectral projection,
        // and Winograd for the wider spatial convolutions on the target GPU.
        const auto preferred = half ? CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            : k == 1 || cin == 3 ? CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM : CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
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
            activation(initial, pixels * channels, half_io());
            activation(state, pixels * channels, half_io());
            activation(norm, pixels * channels, reduced());
            activation(projected, pixels * 2 * hidden, reduced());
            activation(gated, pixels * hidden, reduced());
            const size_t frequency_size = size_t(shape.n) * bins * 2 * fourier;
            activation(spectrum, frequency_size);
            activation(packed, frequency_size, half_io());
            activation(spectral_out, frequency_size, half_io());
            CHECK(cuMemAlloc(&stage_in, shape.rgb_bytes(element(half_io()))));
            CHECK(cuMemAlloc(&stage_out, shape.rgb_bytes(element(half_io())) * scale * scale));
            setup_ms[0] = elapsed(start);

            start = Clock::now();
            const int h = shape.h, w = shape.w;
            make_conv(h, w, 3, channels, 3, reduced());
            make_conv(h, w, channels, 2 * hidden, 3, reduced());
            make_conv(h, w, hidden, channels, 3, reduced());
            make_conv(h, w, channels, 2 * channels, 3, reduced());
            make_conv(h, w, 2 * channels, channels, 3, reduced());
            make_conv(h, w, channels, 3 * scale * scale, 3, reduced());
            make_conv(h, w / 2 + 1, 2 * fourier, 2 * fourier, 1, half_io());
            if (grouped_half_spectral_conv())
                make_conv(h, w / 2 + 1, 2 * fourier, 2 * fourier, 3, true, 2 * fourier);
            if (cudnn_workspace) CHECK(cuMemAlloc(&cudnn_scratch, cudnn_workspace));
            forward_fft.setup(*fft, shape, r2c);
            inverse_fft.setup(*fft, shape, c2r);
            bind_stream(stream);
            setup_ms[1] = elapsed(start);

            start = Clock::now();
            CHECK(cuMemsetD8Async(stage_in, 0, shape.rgb_bytes(element(half_io())), stream));
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
        return base + uint64_t(channel) * shape.pixels() * element(reduced());
    }

    void conv(CUstream target, CUdeviceptr out, CUdeviceptr in, int h, int w, int cin, int cout, int k, Filter filter,
              bool half) {
        const auto& p = *plans.at(Key{h, w, cin, cout, k});
        const float one = 1.f, zero = 0.f;
        CHECK(cudnnConvolutionForward(cudnn, &one, p.input, reinterpret_cast<void*>(in), p.filter,
            reinterpret_cast<void*>(weight(filter.weight, half)), p.convolution, p.algorithm,
            reinterpret_cast<void*>(cudnn_scratch), p.workspace, &zero, p.output, reinterpret_cast<void*>(out)));
        launch(target, half ? BiasHalf : Bias, shape.n * h * w * cout, out, weight(filter.bias, half), cout, h * w);
    }

    void rms(CUstream target, Kernel kernel, CUdeviceptr out, CUdeviceptr in, Norm params,
             int c, int pixels, int out_stride) {
        const bool half = kernel == RmsHalf;
        launch(target, kernel, shape.n * pixels, out, in, weight(params.scale, half), weight(params.offset, half),
               c, c, out_stride, pixels, float(1.0 / std::sqrt(double(c))));
    }

    void fourier_unit(CUstream target, const Fourier& params, CUdeviceptr in, CUdeviceptr out) {
        const int pixels = shape.pixels(), spatial = shape.n * fourier * pixels, frequency = shape.n * fourier * bins;
        const int spectral_channels = 2 * fourier, width = shape.w / 2 + 1;
        // Reuse the gate buffer until the spatial gate runs. It can hold
        // 30 FP32 channels even in reduced precision (128 FP16 channels).
        static_assert(fourier * sizeof(float) <= hidden * sizeof(uint16_t));
        launch(target, reduced() ? CopyChannelsToFloat : CopyChannels, spatial, gated, in, fourier, 2 * hidden, pixels);
        FFT_CHECK(fft->r2c(forward_fft.handle, reinterpret_cast<void*>(gated), reinterpret_cast<void*>(spectrum)));
        launch(target, half_io() ? PackHalf : Pack, frequency, packed, spectrum, fourier, bins, fft_scale);
        rms(target, half_io() ? RmsHalf : Rms, spectral_out, packed, params.pre, spectral_channels, bins, spectral_channels);
        if (half_io()) {
            // Eager uses grouped cuDNN here. Materialize its rounded convolution,
            // bias addition and positional residual before the learned projection.
            if (grouped_half_spectral_conv()) {
                conv(target, packed, spectral_out, shape.h, width, spectral_channels, spectral_channels, 3, params.position, true);
            } else {
                launch(target, DepthwiseHalf, frequency * 2, packed, spectral_out, weight(params.position.weight, true),
                       weight(params.position.bias, true), spectral_channels, spectral_channels, spectral_channels,
                       shape.h, width, 3, 3, 0);
            }
            launch(target, AddHalfResidual, frequency * 2, packed, spectral_out);
        } else {
            launch(target, Depthwise, frequency * 2, packed, spectral_out, weight(params.position.weight, false),
                   weight(params.position.bias, false), spectral_channels, spectral_channels, spectral_channels,
                   shape.h, width, 3, 3, 1);
        }
        conv(target, spectral_out, packed, shape.h, width, spectral_channels, spectral_channels, 1, params.project, half_io());
        launch(target, half_io() ? UnpackHalf : Unpack, frequency, spectrum, spectral_out, bins);
        FFT_CHECK(fft->c2r(inverse_fft.handle, reinterpret_cast<void*>(spectrum), reinterpret_cast<void*>(gated)));
        if (half_io()) {
            launch(target, ScaleFftToHalf, spatial, packed, gated, fft_scale);
            rms(target, RmsHalf, out, packed, params.post, fourier, pixels, channels);
        } else {
            launch(target, ScaleFft, spatial, gated, fft_scale);
            rms(target, mixed() ? RmsToHalf : Rms, out, gated, params.post, fourier, pixels, channels);
        }
    }

    void forward(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const int pixels = shape.pixels(), area = shape.n * pixels, h = shape.h, w = shape.w;
        const bool half = reduced();
        if (mixed()) {
            launch(target, ToHalf, area * 3, gated, input);
            conv(target, projected, gated, h, w, 3, channels, 3, model.first, true);
            launch(target, ToFloat, area * channels, initial, projected);
        } else {
            conv(target, initial, input, h, w, 3, channels, 3, model.first, half);
        }
        CHECK(cuMemcpyDtoDAsync(state, initial, uint64_t(area) * channels * element(half_io()), target));
        const Kernel spatial_norm = mixed() ? RmsToHalf : half_io() ? RmsHalf : Rms;
        const Kernel spatial_gate = mixed() ? GateMixed : half_io() ? GateHalf : Gate;
        const Kernel spatial_residual = mixed() ? ResidualMixed : half_io() ? ResidualHalf : Residual;
        for (int i = 0; i < blocks; ++i) {
            const auto& block = model.body[i];
            rms(target, spatial_norm, norm, state, block.norm, channels, pixels, channels);
            conv(target, projected, norm, h, w, channels, 2 * hidden, 3, block.expand, half);
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
                    launch(target, half ? DepthwiseHalf : Depthwise, area * band, out, in, weight(filter.weight, half),
                           weight(filter.bias, half), band, 2 * hidden, channels, h, w, kh, kw, 0);
                    offset += band;
                }
            }
            launch(target, spatial_gate, area * hidden, gated, projected, norm, hidden, channels, pixels);
            conv(target, projected, gated, h, w, hidden, channels, 3, block.project, half);
            launch(target, spatial_residual, area * channels, state, projected, weight(block.gamma, half_io()), channels, pixels);
        }
        if (mixed()) launch(target, ToHalf, area * channels, gated, state);
        conv(target, projected, mixed() ? gated : state, h, w, channels, 2 * channels, 3, model.tail_expand, half);
        launch(target, half ? SiluHalf : Silu, area * 2 * channels, projected);
        conv(target, norm, projected, h, w, 2 * channels, channels, 3, model.tail_project, half);
        launch(target, mixed() ? AddMixed : half ? AddHalf : Add, area * channels, state, norm, initial);
        if (mixed()) launch(target, ToHalf, area * channels, gated, state);
        conv(target, projected, mixed() ? gated : state, h, w, channels, 3 * scale * scale, 3, model.last, half);
        launch(target, mixed() ? ShuffleMixed : half ? ShuffleHalf : Shuffle, area * 3 * scale * scale, output, projected, h, w);
    }

    void execute(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        if (!executable) {
            bind_stream(target);
            forward(input, output, target);
            return;
        }
        const auto bytes = shape.rgb_bytes(element(half_io()));
        if (input != stage_in) CHECK(cuMemcpyDtoDAsync(stage_in, input, bytes, target));
        CHECK(cuGraphLaunch(executable, target));
        if (output != stage_out) CHECK(cuMemcpyDtoDAsync(output, stage_out, bytes * scale * scale, target));
    }

public:
    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "gfisrv2", scale, 0}; }
    UmziIo tensors() const override {
        const uint32_t dtype = half_io() ? UMZI_FLOAT16 : UMZI_FLOAT32;
        return {sizeof(UmziIo), {"input", {4, {0, 3, 0, 0}}, dtype, 0},
                                {"output", {4, {0, 3, 0, 0}}, dtype, 0}};
    }

    UmziIo describe(const UmziShape& request) const override {
        const auto concrete = checked(request);
        auto io = tensors();
        io.input.shape = request;
        io.input.bytes = concrete.rgb_bytes(element(half_io()));
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
        const auto bytes = shape.rgb_bytes(element(half_io()));
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
        info.buffer_bytes[0] = (weights ? model.bytes : 0) + (weights_half ? model.bytes / 2 : 0);
        if (shape.n) {
            info.prepared_input = shape.nchw();
            info.buffer_bytes[1] = activation_bytes;
            info.buffer_bytes[2] = cudnn_workspace + forward_fft.bytes + inverse_fft.bytes;
            info.buffer_bytes[3] = shape.rgb_bytes(element(half_io())) * (1 + scale * scale);
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
    const char* kernel_message() const override {
        if (mixed()) return "GFISRv2 mixed precision with FP32 Fourier and residuals";
        if (half_io()) return "GFISRv2 FP16 with FP32 cuFFT";
        return "GFISRv2 FP32 with cuFFT";
    }
};
} // namespace

std::unique_ptr<umzi::KernelSession> umzi::make_gfisrv2(const ModelPackage& package, const UmziSessionOptions& options) {
    auto session = std::make_unique<Session>();
    session->open(package, options);
    return session;
}
