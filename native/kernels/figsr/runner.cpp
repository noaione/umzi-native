#include "../../runtime/binary.h"
#include "../../runtime/kernel.h"
#include "cudnn_conv_plan.h"
#include "cufft_api.h"
#include "figsr_images.h"

#include <climits>
#include <cmath>
#include <map>
#include <numeric>

namespace {
using umzi::ContextScope;
#define CHECK(call) umzi::check((call), #call)

constexpr int scale = 2, border = 4, features = 32, hidden = 80, fourier = 8, group_blocks = 12;
constexpr int compact_expansion = 7 * fourier;
constexpr int r2c = 0x2a, c2r = 0x2c;
constexpr std::array<uint32_t, 11> configuration{3, 3, scale, border, 2, features, group_blocks, hidden, fourier, 13, 17};

void check_fft(int status, const char* operation) {
    if (status == 2) throw umzi::AllocationError(std::string(operation) + ": cuFFT allocation failed");
    if (status) throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(status));
}
#define FFT_CHECK(call) check_fft((call), #call)

struct Shape {
    int n = 0, h = 0, w = 0;
    bool operator==(const Shape& other) const { return n == other.n && h == other.h && w == other.w; }
    int area() const { return h * w; }
    int pixels() const { return n * area(); }
    UmziShape nchw() const { return {4, {uint32_t(n), 3, uint32_t(h), uint32_t(w)}}; }
    Shape padded() const { return {n, h + 2 * border + h % 2, w + 2 * border + w % 2}; }
};

Shape checked(const UmziShape& shape) {
    const auto* d = shape.dimensions;
    const uint64_t n = d[0], h = d[2], w = d[3];
    if (shape.rank != 4 || !n || d[1] != 3 || h < 6 || w < 6 || h > INT_MAX - 9 || w > INT_MAX - 9 ||
        (h + 8 + h % 2) * (w + 8 + w % 2) > uint64_t(INT_MAX / (2 * hidden)) / n)
        throw std::runtime_error("FiGSR expects a supported NCHW RGB shape with height and width at least 6");
    return {int(n), int(h), int(w)};
}

struct Filter {
    int cin, cout, kh, kw, groups = 1;
    size_t weight = 0, bias = 0;
};

struct Block {
    size_t norm = 0, spectral_norm = 0, inverse_norm = 0;
    Filter expand{features, 2 * hidden, 3, 3}, fpe{2 * fourier, 2 * fourier, 3, 3, 2 * fourier};
    Filter fdc{2 * fourier, 2 * fourier, 1, 1}, square{fourier, fourier, 13, 13};
    Filter row{fourier, fourier, 1, 17}, column{fourier, fourier, 17, 1}, project{hidden, features, 3, 3};
};

struct Network {
    struct Parameter { size_t offset, count; bool filter; };
    std::vector<Parameter> parameters;
    size_t bytes = 0, filter_bytes, rgb;
    Filter head{8, features, 3, 3}, end{features, features, 3, 3};
    Filter combine{3 * features, features, 1, 1}, tail{features, 16, 3, 3};
    std::array<Block, 2 * group_blocks> blocks{};

    explicit Network(size_t element_bytes = sizeof(float)) : filter_bytes(element_bytes), rgb(pack(6)) {
        pack(head);
        for (auto& block : blocks) {
            block.norm = norm(features);
            block.spectral_norm = norm(2 * fourier);
            block.inverse_norm = norm(fourier);
            for (auto* filter : {&block.expand, &block.fpe, &block.fdc, &block.square, &block.row,
                                 &block.column, &block.project}) pack(*filter);
        }
        pack(end);
        pack(combine);
        pack(tail);
    }

    size_t pack(size_t count, bool filter = false) {
        const auto offset = (bytes + 255) & ~size_t(255);
        bytes = offset + count * (filter ? filter_bytes : sizeof(float));
        parameters.push_back({offset, count, filter});
        return offset;
    }

    size_t norm(int channels) { return pack(2 * channels + 2); }

    void pack(Filter& filter) {
        filter.weight = pack(size_t(filter.cout) * (filter.cin / filter.groups) * filter.kh * filter.kw, true);
        filter.bias = pack(filter.cout);
    }

    std::vector<char> half_weights(const Network& source, std::string_view weights) const {
        std::vector<char> packed(bytes);
        for (size_t i = 0; i < parameters.size(); ++i) {
            const auto& parameter = parameters[i];
            const char* input = weights.data() + source.parameters[i].offset;
            char* output = packed.data() + parameter.offset;
            if (!parameter.filter) {
                std::memcpy(output, input, parameter.count * sizeof(float));
                continue;
            }
            for (size_t j = 0; j < parameter.count; ++j) {
                float value;
                std::memcpy(&value, input + j * sizeof(value), sizeof(value));
                const auto half = umzi::float_to_half(value);
                if ((half & 0x7fff) >= 0x7c00) throw std::runtime_error("FiGSR filter exceeds the finite FP16 range");
                std::memcpy(output + j * sizeof(half), &half, sizeof(half));
            }
        }
        return packed;
    }

    std::vector<char> nchw_body_weights(std::string_view source) const {
        std::vector<char> packed(source.begin(), source.end());
        for (const auto& block : blocks) {
            for (const auto* filter : {&block.expand, &block.project}) {
                const int taps = filter->kh * filter->kw;
                for (int out = 0; out < filter->cout; ++out) {
                    for (int in = 0; in < filter->cin; ++in) {
                        for (int tap = 0; tap < taps; ++tap) {
                            const auto ohwi = (out * taps + tap) * filter->cin + in;
                            const auto oihw = (out * filter->cin + in) * taps + tap;
                            std::memcpy(packed.data() + filter->weight + oihw * sizeof(float),
                                        source.data() + filter->weight + ohwi * sizeof(float), sizeof(float));
                        }
                    }
                }
            }
        }
        return packed;
    }
};

enum Kernel { Pack, Bias, Rms, ResidualNorm, Copy, Extract, SplitSpatial, GateSpatial,
              Spectrum, Gelu, Inverse, GateIdentity, Shuffle,
              RmsNchw, ResidualNormNchw, ExtractNchw, SplitSpatialNchw, GateSpatialNchw, InverseNchw,
              GateIdentityNchw, ExtractHalf, RowConvGate, ExpandGateHalf, KernelCount };
constexpr std::array<const char*, KernelCount> names{
    "pack_rgb", "bias_residual", "rms_norm", "residual_norm", "copy_channels", "extract_fourier",
    "split_spatial", "gate_spatial", "spectrum_norm", "spectrum_gelu", "inverse_norm", "gate_identity", "shuffle_rgb",
    "rms_norm_nchw", "residual_norm_nchw", "extract_fourier_nchw", "split_spatial_nchw", "gate_spatial_nchw",
    "inverse_norm_nchw", "gate_identity_nchw", "extract_fourier_half", "row_conv_gate", "expand_gate_half"};
enum Buffer { State, Normalized, Stem, Middle, Expanded, Values, Fourier, Complex, Spectral, SpectralWork, BufferCount };

class Session final : public umzi::KernelSession {
    struct Shared {
        CUdevice device{};
        CUcontext context = nullptr;
        CUmodule module = nullptr;
        cudnnHandle_t cudnn = nullptr;
        std::shared_ptr<umzi::CufftApi> fft;
        CUdeviceptr weights = 0;

        ~Shared() {
            if (!context) return;
            const umzi::PrimaryContextRelease release{device};
            if (cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
            if (cudnn) cudnnDestroy(cudnn);
            if (weights) cuMemFree(weights);
            if (module) cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    };
    std::shared_ptr<Shared> shared = std::make_shared<Shared>();

    Network net;
    bool fp16 = false, fp32 = false, graph = false, pending = false;
    bool tune_convolutions = false, fp32_nchw = false, fp16_expand_accumulation = false;
    bool fuse_row_gate = false, fuse_expand_gate = false;
    CUdevice device{};
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUstream stream = nullptr;
    CUevent complete = nullptr;
    std::array<CUevent, 2> tuning_events{};
    cudnnHandle_t cudnn = nullptr;
    // Clones own their handle: it carries its stream, so sessions on different streams cannot share one.
    bool own_cudnn = false;
    std::shared_ptr<umzi::CufftApi> fft;
    std::array<int, 2> fft_plans{};
    std::array<bool, 2> fft_created{};
    std::array<CUfunction, KernelCount> functions{};
    CUgraphExec executable = nullptr;
    CUdeviceptr weights = 0, stage_in = 0, stage_out = 0, conv_workspace = 0, fft_workspace = 0;
    std::array<CUdeviceptr, BufferCount> buffers{};
    Shape shape;
    size_t activation_bytes = 0, conv_bytes = 0, fft_bytes = 0;
    int32_t fp32_fallbacks = 0;
    int32_t timed_candidates = 0, changed_plans = 0;
    std::array<double, 4> setup_ms{};
    std::array<uint64_t, 4> device_used{};
    using PlanKey = std::array<int, 7>;
    std::map<PlanKey, umzi::CudnnBackendPlan> plans;
    std::map<std::array<CUdeviceptr, 3>, std::unique_ptr<umzi::CudnnDescriptor>> variants;

    size_t rgb_bytes(Shape input) const { return size_t(input.pixels()) * 3 * (fp16 ? 2 : 4); }
    size_t element_bytes() const { return fp32 ? sizeof(float) : sizeof(uint16_t); }

    std::array<size_t, BufferCount> activation_sizes(Shape padded) const {
        const Shape frequency{padded.n, padded.h, padded.w / 2 + 1};
        const size_t image = size_t(padded.pixels()) * element_bytes();
        const size_t spectrum = size_t(frequency.pixels()) * element_bytes();
        const size_t fft_input = size_t(padded.pixels()) * fourier * sizeof(float);
        const size_t fft_output = size_t(frequency.pixels()) * fourier * 2 * sizeof(float);
        return {
            features * image, features * image, features * image, features * image,
            fuse_expand_gate ? (compact_expansion + hidden) * image : 2 * hidden * image,
            fuse_expand_gate ? size_t(0) : hidden * image,
            fft_input, fft_output, 2 * fourier * spectrum, 2 * fourier * spectrum,
        };
    }

    void ensure_staging() {
        if (stage_in && stage_out) return;
        const size_t input_bytes = rgb_bytes(shape);
        const size_t needed = (stage_out ? 0 : input_bytes * scale * scale) + (stage_in ? 0 : input_bytes);
        umzi::require_device_memory(needed, "FiGSR staging buffers", device);
        // Allocate only what is missing: a failed second allocation must not leave a half-built pair that passes the early return.
        if (!stage_out) CHECK(cuMemAlloc(&stage_out, input_bytes * scale * scale));
        if (!stage_in) CHECK(cuMemAlloc(&stage_in, input_bytes));
    }

    void wait() {
        if (!pending) return;
        CHECK(cuEventSynchronize(complete));
        pending = false;
    }

    void release_shape() {
        if (executable) cuGraphExecDestroy(executable);
        executable = nullptr;
        variants.clear();
        plans.clear();
        for (size_t i = 0; i < fft_plans.size(); ++i) {
            if (fft_created[i]) fft->destroy(fft_plans[i]);
            fft_created[i] = false;
        }
        for (auto* pointer : {&stage_in, &stage_out, &conv_workspace, &fft_workspace}) {
            if (*pointer) cuMemFree(*pointer);
            *pointer = 0;
        }
        for (size_t i = 0; i < BufferCount; ++i) {
            if (fuse_expand_gate && i == Values) {
                buffers[i] = 0;
                continue;
            }
            if (buffers[i]) cuMemFree(buffers[i]);
            buffers[i] = 0;
        }
        shape = {};
        activation_bytes = conv_bytes = fft_bytes = 0;
        fp32_fallbacks = 0;
        timed_candidates = changed_plans = 0;
        setup_ms = {};
        device_used = {};
    }

    void prepare_fft(Shape padded) {
        constexpr int types[] = {r2c, c2r};
        for (size_t i = 0; i < fft_plans.size(); ++i) {
            FFT_CHECK(fft->create(&fft_plans[i]));
            fft_created[i] = true;
            FFT_CHECK(fft->auto_allocation(fft_plans[i], 0));
            int dimensions[] = {padded.h, padded.w};
            size_t bytes = 0;
            FFT_CHECK(fft->make_plan(fft_plans[i], 2, dimensions, nullptr, 1, 0, nullptr, 1, 0, types[i],
                                     padded.n * fourier, &bytes));
            fft_bytes = std::max(fft_bytes, bytes);
        }
        if (fft_bytes) {
            umzi::allocate_device_checked(fft_workspace, fft_bytes, "FiGSR cuFFT workspace", device);
            for (const auto handle : fft_plans)
                FFT_CHECK(fft->work_area(handle, reinterpret_cast<void*>(fft_workspace)));
        }
    }

    static PlanKey key(Shape shape, const Filter& filter) {
        return {shape.h, shape.w, filter.cin, filter.cout, filter.kh, filter.kw, filter.groups};
    }

    void prepare_conv(Shape shape, const Filter& filter, size_t limit) {
        const auto index = key(shape, filter);
        if (plans.find(index) != plans.end()) return;
        const auto& body = net.blocks.front();
        const auto layout = fp32_nchw && (&filter == &body.expand || &filter == &body.project)
            ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
        const bool prefer_half = fp16 || (fp16_expand_accumulation && &filter == &body.expand);
        umzi::CudnnBackendPlan selected;
        for (bool half_compute : {true, false}) {
            if (half_compute && !prefer_half) continue;
            selected.graph = std::make_unique<umzi::CudnnConvGraph>(
                cudnn, shape.n, shape.h, shape.w, filter.cin, filter.cout, filter.kh, filter.kw, filter.groups,
                fp32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF, half_compute, layout);
            auto candidates = selected.graph->candidates(cudnn, limit, true);
            if (candidates.empty()) continue;
            selected.workspace = size_t(candidates[0].workspace);
            selected.plan = std::move(candidates[0].plan);
            conv_bytes = std::max(conv_bytes, selected.workspace);
            if (prefer_half && !half_compute) ++fp32_fallbacks;
            plans.emplace(index, std::move(selected));
            return;
        }
        throw std::runtime_error("No cuDNN FiGSR plan for " + std::to_string(filter.cin) + " to " +
                                 std::to_string(filter.cout) + " channels, " + std::to_string(filter.kh) + "x" +
                                 std::to_string(filter.kw) + " kernel");
    }

    void conv(CUdeviceptr out, CUdeviceptr in, const Filter& filter, Shape grid) {
        const std::array binding{in, weights + filter.weight, out};
        auto bound = variants.find(binding);
        if (bound == variants.end()) {
            auto variant = umzi::bind_conv_variant(reinterpret_cast<void*>(in),
                reinterpret_cast<void*>(weights + filter.weight), reinterpret_cast<void*>(out),
                reinterpret_cast<void*>(conv_workspace));
            bound = variants.emplace(binding, std::move(variant)).first;
        }
        CHECK(cudnnBackendExecute(cudnn, plans.at(key(grid, filter)).plan->get(), bound->second->get()));
    }

    void tune_conv(Shape padded, const Filter& filter, Buffer input, Buffer output) {
        auto& selected = plans.at(key(padded, filter));
        auto candidates = selected.graph->candidates(cudnn, conv_bytes);
        if (candidates.size() < 2) return;
        CHECK(cuMemsetD8Async(buffers[input], 0, size_t(padded.pixels()) * filter.cin * element_bytes(), stream));
        const auto elapsed = [&](const umzi::CudnnBackendPlan& plan, float best) {
            ++timed_candidates;
            // Pass the member size so it stays in step with conv_workspace if the scratch is ever regrown.
            return umzi::time_cudnn_backend_plan(cudnn, stream, tuning_events, plan, buffers[input],
                weights + filter.weight, buffers[output], conv_workspace, conv_bytes, best);
        };
        const auto control = elapsed(selected, std::numeric_limits<float>::infinity());
        float fastest = control;
        umzi::CudnnBackendPlan winner;
        for (auto& candidate : candidates) {
            umzi::CudnnBackendPlan trial;
            trial.plan = std::move(candidate.plan);
            trial.workspace = size_t(candidate.workspace);
            const auto elapsed_ms = elapsed(trial, fastest);
            if (elapsed_ms < fastest) {
                fastest = elapsed_ms;
                winner = std::move(trial);
            }
        }
        if (winner.plan && fastest < 0.97f * control) {
            selected.plan = std::move(winner.plan);
            selected.workspace = winner.workspace;
            ++changed_plans;
        }
    }

    template<class... Args>
    void launch(CUstream target, Kernel kernel, int count, Args... args) {
        constexpr unsigned threads = 256;
        void* parameters[] = {&args...};
        CHECK(cuLaunchKernel(functions[kernel], (unsigned(count) + threads - 1) / threads, 1, 1,
                             threads, 1, 1, 0, target, parameters, nullptr));
    }

    template<class... Args>
    void launch_body(CUstream target, Kernel nhwc, Kernel nchw, int count, int area, Args... args) {
        if (fp32_nchw) launch(target, nchw, count, args..., area);
        else launch(target, nhwc, count, args..., fp16, fp32);
    }

    void bind_stream(CUstream target) {
        CHECK(cudnnSetStream(cudnn, target));
        for (const auto handle : fft_plans) FFT_CHECK(fft->set_stream(handle, reinterpret_cast<void*>(target)));
    }

    void forward(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const Shape spatial = shape.padded(), frequency{spatial.n, spatial.h, spatial.w / 2 + 1};
        const int area = spatial.area(), pixels = spatial.pixels(), bins = frequency.area(),
                  frequency_pixels = frequency.pixels();
        const float ortho = 1.f / std::sqrt(float(area));
        const auto& b = buffers;
        const auto copy = [&](Buffer out, Buffer in, int out_stride, int in_stride, int out_offset, int in_offset, int width) {
            const int count = pixels * width;
            launch(target, Copy, count, b[out], b[in], out_stride, in_stride, out_offset, in_offset, width, count, fp32);
        };
        const auto biased = [&](Buffer out, Buffer in, const Filter& filter) {
            conv(b[out], b[in], filter, spatial);
            const int count = pixels * filter.cout;
            launch(target, Bias, count, b[out], b[out], weights + filter.bias, CUdeviceptr(0), filter.cout, count, fp16, fp32);
        };
        // The spectral buffer holds at least eight spatial lanes per pixel.
        launch(target, Pack, pixels * 8, b[Spectral], input, weights + net.rgb, shape.h, shape.w,
               spatial.h, spatial.w, pixels * 8, fp16, fp32);
        biased(State, Spectral, net.head);
        const size_t saved_bytes = size_t(pixels) * features * element_bytes();
        CHECK(cuMemcpyDtoDAsync(b[Stem], b[State], saved_bytes, target));
        launch_body(target, Rms, RmsNchw, features * pixels, area, b[Normalized], b[State],
                    weights + net.blocks.front().norm, pixels);
        for (size_t i = 0; i < net.blocks.size(); ++i) {
            const auto& block = net.blocks[i];
            const auto expand_bias = weights + block.expand.bias;
            if (fuse_expand_gate) {
                const int tiles_x = (spatial.w + 15) / 16, tiles_y = (spatial.h + 3) / 4;
                launch(target, ExpandGateHalf, spatial.n * tiles_x * tiles_y * 256, b[Expanded], b[Values], b[Fourier], b[Normalized],
                       weights + block.expand.weight, expand_bias, spatial.h, spatial.w, tiles_x, tiles_y,
                       fp16 || fp16_expand_accumulation, fp16);
            } else {
                conv(b[Expanded], b[Normalized], block.expand, spatial);
                const int plain = pixels * (hidden - 4 * fourier);
                launch_body(target, GateIdentity, GateIdentityNchw, plain, area, b[Values], b[Expanded], expand_bias, plain);
            }

            const int fourier_count = pixels * fourier;
            if (!fuse_expand_gate) {
                if (fp32_nchw)
                    launch(target, ExtractNchw, fourier_count, b[Fourier], b[Expanded], expand_bias, area, fourier_count);
                else if (!fp32)
                    launch(target, ExtractHalf, pixels, b[Fourier], b[Expanded], expand_bias, area, pixels, fp16);
                else
                    launch(target, Extract, fourier_count, b[Fourier], b[Expanded], expand_bias, area, fourier_count, fp16, fp32);
            }
            FFT_CHECK(fft->r2c(fft_plans[0], reinterpret_cast<void*>(b[Fourier]), reinterpret_cast<void*>(b[Complex])));
            launch(target, Spectrum, 2 * fourier * frequency_pixels, b[Spectral], b[Complex], weights + block.spectral_norm,
                   bins, frequency_pixels, ortho, fp16, fp32);
            conv(b[SpectralWork], b[Spectral], block.fpe, frequency);
            const int spectrum_elements = frequency_pixels * 2 * fourier;
            launch(target, Bias, spectrum_elements, b[SpectralWork], b[SpectralWork], weights + block.fpe.bias,
                   b[Spectral], 2 * fourier, spectrum_elements, fp16, fp32);
            conv(b[Spectral], b[SpectralWork], block.fdc, frequency);
            launch(target, Gelu, frequency_pixels * fourier, b[Complex], b[Spectral], weights + block.fdc.bias,
                   bins, frequency_pixels * fourier, fp16, fp32);
            FFT_CHECK(fft->c2r(fft_plans[1], reinterpret_cast<void*>(b[Complex]), reinterpret_cast<void*>(b[Fourier])));
            const auto inverse_norm = weights + block.inverse_norm;
            if (fp32_nchw)
                launch(target, InverseNchw, fourier_count, b[Values], b[Fourier], inverse_norm, b[Expanded], expand_bias,
                       area, pixels, ortho);
            else
                launch(target, Inverse, fourier_count, b[Values], b[Fourier], inverse_norm, b[Expanded], expand_bias,
                       area, pixels, ortho, fp16, fp32, fuse_expand_gate);

            int offset = hidden - 3 * fourier;
            // Both Fourier staging buffers are dead after the inverse gate.
            for (const auto* filter : {&block.square, &block.row, &block.column}) {
                const int count = pixels * fourier;
                if (fp32_nchw)
                    launch(target, SplitSpatialNchw, count, b[Spectral], b[Expanded], expand_bias, offset, count, area);
                else
                    launch(target, SplitSpatial, count, b[Spectral], b[Expanded], expand_bias, offset, count, fp16, fp32,
                           fuse_expand_gate);
                if (fuse_row_gate && filter == &block.row) {
                    const int threads = 32 * ((spatial.w + 15) / 16) * spatial.h * spatial.n;
                    launch(target, RowConvGate, threads, b[Values], b[Spectral], weights + filter->weight,
                           weights + filter->bias, b[Expanded], expand_bias, spatial.w, spatial.h * spatial.n, offset,
                           fuse_expand_gate);
                } else {
                    conv(b[Fourier], b[Spectral], *filter, spatial);
                    if (fp32_nchw)
                        launch(target, GateSpatialNchw, count, b[Values], b[Fourier], weights + filter->bias,
                               b[Expanded], expand_bias, offset, count, area);
                    else
                        launch(target, GateSpatial, count, b[Values], b[Fourier], weights + filter->bias,
                               b[Expanded], expand_bias, offset, count, fp16, fp32, fuse_expand_gate);
                }
                offset += fourier;
            }
            conv(b[Normalized], b[Values], block.project, spatial);
            if (fp32_nchw) {
                const auto parameters = i + 1 < net.blocks.size() ? weights + net.blocks[i + 1].norm : CUdeviceptr(0);
                launch(target, ResidualNormNchw, pixels * 8, b[State], b[Normalized], weights + block.project.bias,
                       parameters, pixels, area);
            } else if (i + 1 < net.blocks.size())
                launch(target, ResidualNorm, pixels * features, b[State], b[Normalized], weights + block.project.bias,
                       weights + net.blocks[i + 1].norm, pixels, fp16, fp32);
            else
                launch(target, Bias, pixels * features, b[State], b[Normalized], weights + block.project.bias,
                       b[State], features, pixels * features, fp16, fp32);
            if (i + 1 == group_blocks)
                CHECK(cuMemcpyDtoDAsync(b[Middle], b[State], saved_bytes, target));
        }
        biased(Normalized, State, net.end);
        copy(Expanded, Normalized, 3 * features, features, 0, 0, features);
        copy(Expanded, Stem, 3 * features, features, features, 0, features);
        copy(Expanded, Middle, 3 * features, features, 2 * features, 0, features);
        biased(State, Expanded, net.combine);
        conv(b[Normalized], b[State], net.tail, spatial);
        const int count = shape.pixels() * 3 * scale * scale;
        launch(target, Shuffle, count, output, b[Normalized], weights + net.tail.bias, weights + net.rgb,
               shape.h, shape.w, spatial.h, spatial.w, count, fp16, fp32);
    }

    void prepare(Shape requested) {
        if (shape == requested) return;
        wait(); // Only a shape change frees buffers the previous batch may still use.
        release_shape();
        try {
            const auto padded = requested.padded();
            const Shape frequency{padded.n, padded.h, padded.w / 2 + 1};
            const size_t image = size_t(padded.pixels()) * element_bytes();
            const auto bytes = activation_sizes(padded);
            size_t total_act = 0;
            for (size_t value : bytes) total_act += value;
            const size_t stage_bytes = graph ? rgb_bytes(requested) * (1 + scale * scale) : 0;
            auto start = umzi::Clock::now();
            device_used[0] = umzi::device_used_bytes();
            umzi::require_device_memory(total_act + stage_bytes, "FiGSR shape buffers", device);
            for (size_t i = 0; i < bytes.size(); ++i) {
                if (!bytes[i]) continue;
                CHECK(cuMemAlloc(&buffers[i], bytes[i]));
                activation_bytes += bytes[i];
            }
            if (fuse_expand_gate) buffers[Values] = buffers[Expanded] + compact_expansion * image;
            device_used[1] = umzi::device_used_bytes();
            setup_ms[0] = umzi::elapsed_ms(start);

            start = umzi::Clock::now();
            prepare_fft(padded);
            bind_stream(stream);
            const size_t available = umzi::device_vram_info(device).free_bytes;
            const auto cap = std::min<size_t>(available / 2, 256u << 20);
            for (const auto* filter : {&net.head, &net.end, &net.combine, &net.tail}) prepare_conv(padded, *filter, cap);
            const auto& block = net.blocks.front();
            for (const auto* filter : {&block.expand, &block.square, &block.row, &block.column, &block.project})
                if ((!fuse_row_gate || filter != &block.row) && (!fuse_expand_gate || filter != &block.expand))
                    prepare_conv(padded, *filter, cap);
            prepare_conv(frequency, block.fpe, cap);
            prepare_conv(frequency, block.fdc, cap);
            if (conv_bytes) umzi::allocate_device_checked(conv_workspace, conv_bytes, "FiGSR cuDNN workspace", device);
            if (tune_convolutions) {
                if (!fuse_expand_gate) tune_conv(padded, block.expand, Normalized, Expanded);
                tune_conv(padded, block.project, Values, Normalized);
                for (const auto* filter : {&block.square, &block.row, &block.column})
                    if (!fuse_row_gate || filter != &block.row) tune_conv(padded, *filter, Spectral, Fourier);
            }
            device_used[2] = umzi::device_used_bytes();
            setup_ms[1] = umzi::elapsed_ms(start);

            shape = requested;
            if (graph) {
                start = umzi::Clock::now();
                ensure_staging();
                CHECK(cuMemsetD8Async(stage_in, 0, rgb_bytes(requested), stream));
                forward(stage_in, stage_out, stream);
                CHECK(cuStreamSynchronize(stream));
                setup_ms[2] = umzi::elapsed_ms(start);
                start = umzi::Clock::now();
                umzi::capture_cuda_graph(stream, executable, [&] { forward(stage_in, stage_out, stream); });
                setup_ms[3] = umzi::elapsed_ms(start);
            }
            device_used[3] = umzi::device_used_bytes();
        } catch (const umzi::AllocationError&) {
            // Out of memory is for the tiler to handle, not a reason to drop fusion.
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        } catch (const std::runtime_error& error) {
            // Only runtime failures may trigger the fallback; bad_alloc and logic errors fall through to the rethrow below.
            cuStreamSynchronize(stream);
            release_shape();
            if (!fuse_row_gate && !fuse_expand_gate) throw;
            fuse_row_gate = fuse_expand_gate = false;
            warn_option(std::string("Disabled FiGSR gate fusion: ") + error.what());
            prepare(requested);
        } catch (...) {
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        }
    }

    void execute(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        if (!executable) {
            bind_stream(target);
            forward(input, output, target);
            return;
        }
        const auto bytes = rgb_bytes(shape);
        if (input != stage_in) CHECK(cuMemcpyDtoDAsync(stage_in, input, bytes, target));
        CHECK(cuGraphLaunch(executable, target));
        if (output != stage_out) CHECK(cuMemcpyDtoDAsync(output, stage_out, bytes * scale * scale, target));
    }

public:
    void open(const umzi::ModelPackage& package, const UmziSessionOptions& options) {
        if (package.schema != 1) throw std::runtime_error("Unsupported FiGSR model payload version");
        umzi::BinaryReader architecture(package.section(1));
        for (const auto expected : configuration)
            if (architecture.number<uint32_t>() != expected) throw std::runtime_error("Unsupported FiGSR architecture");
        const auto data = package.section(3);
        if (architecture.size() || !package.section(2).empty() || package.section(4).size() != 32 || data.size() != net.bytes)
            throw std::runtime_error("Invalid FiGSR metadata or weights");
        fp16 = options.precision == UMZI_FP16;
        fp32 = options.precision == UMZI_FP32;
        bool tune_seen = false, layout_seen = false, expand_seen = false, row_seen = false, fusion_seen = false;
        for (uint32_t i = 0; i < options.kernel_option_count; ++i) {
            const auto& option = options.kernel_options[i];
            const std::string_view name(option.name ? option.name : "");
            if (option.value != 0 && option.value != 1)
                throw std::runtime_error("FiGSR kernel options require a value of 0 or 1");
            if (name == "tune_convolutions" && !tune_seen) {
                tune_seen = true;
                tune_convolutions = option.value != 0;
            } else if (name == "fp32_nchw" && !layout_seen) {
                layout_seen = true;
                fp32_nchw = option.value != 0;
            } else if (name == "fp16_expand_accumulation" && !expand_seen) {
                expand_seen = true;
                fp16_expand_accumulation = option.value != 0;
            } else if (name == "fuse_row_gate" && !row_seen) {
                row_seen = true;
                fuse_row_gate = option.value != 0;
            } else if (name == "fuse_expand_gate" && !fusion_seen) {
                fusion_seen = true;
                fuse_expand_gate = option.value != 0;
            } else {
                throw std::runtime_error("Invalid or duplicate FiGSR kernel option: " + std::string(name));
            }
        }
        if (!fp32 && fp32_nchw) {
            warn_option("Disabled FiGSR fp32_nchw: requires FP32 precision");
            fp32_nchw = false;
        }
        if ((fp16 || fp32) && fp16_expand_accumulation) {
            warn_option("Disabled FiGSR fp16_expand_accumulation: requires mixed precision");
            fp16_expand_accumulation = false;
        }
        if ((fp16 || fp32) && fuse_row_gate) {
            warn_option("Disabled FiGSR fuse_row_gate: requires mixed precision");
            fuse_row_gate = false;
        }
        if (fp32 && fuse_expand_gate) {
            warn_option("Disabled FiGSR fuse_expand_gate: requires mixed or FP16 precision");
            fuse_expand_gate = false;
        }
        if (!fp32) net = Network(sizeof(uint16_t));
        graph = (options.flags & UMZI_CUDA_GRAPH) != 0;
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        shared->device = device;
        shared->context = context;
        ContextScope scope(context);
        fft = std::make_shared<umzi::CufftApi>();
        shared->fft = fft;
        umzi::load_kernel_module(device, module, figsr_images::embedded_cubins, figsr_images::embedded_ptx);
        shared->module = module;
        for (size_t i = 0; i < names.size(); ++i) CHECK(cuModuleGetFunction(&functions[i], module, names[i]));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        if (tune_convolutions)
            for (auto& event : tuning_events) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        CHECK(cudnnCreate(&cudnn));
        shared->cudnn = cudnn;
        CHECK(cuMemAlloc(&weights, net.bytes));
        shared->weights = weights;
        if (fp32_nchw) {
            const auto packed = net.nchw_body_weights(data);
            CHECK(cuMemcpyHtoD(weights, packed.data(), packed.size()));
        } else if (fp32) {
            CHECK(cuMemcpyHtoD(weights, data.data(), data.size()));
        } else {
            const auto packed = net.half_weights(Network{}, data);
            CHECK(cuMemcpyHtoD(weights, packed.data(), packed.size()));
        }
    }

    ~Session() override {
        if (context && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            if (pending) cuEventSynchronize(complete);
            if (stream) cuStreamSynchronize(stream);
            release_shape();
            if (own_cudnn && cudnn) cudnnDestroy(cudnn);
            if (complete) cuEventDestroy(complete);
            for (auto event : tuning_events) if (event) cuEventDestroy(event);
            if (stream) cuStreamDestroy(stream);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        fft.reset();
        shared.reset();
    }

    std::unique_ptr<umzi::KernelSession> clone_shared() const override {
        auto copy = std::make_unique<Session>();
        copy->copy_base_state(*this);
        copy->shared = shared;
        copy->net = net;
        copy->fp16 = fp16;
        copy->fp32 = fp32;
        copy->graph = graph;
        copy->tune_convolutions = tune_convolutions;
        copy->fp32_nchw = fp32_nchw;
        copy->fp16_expand_accumulation = fp16_expand_accumulation;
        copy->fuse_row_gate = fuse_row_gate;
        copy->fuse_expand_gate = fuse_expand_gate;
        copy->device = device;
        copy->context = context;
        copy->module = module;
        copy->fft = fft;
        copy->functions = functions;
        copy->weights = weights;
        ContextScope scope(context);
        copy->own_cudnn = true;
        CHECK(cudnnCreate(&copy->cudnn));
        CHECK(cuStreamCreate(&copy->stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&copy->complete, CU_EVENT_DISABLE_TIMING));
        if (tune_convolutions)
            for (auto& event : copy->tuning_events) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        return copy;
    }

    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "figsr", scale, 0}; }
    umzi::TilingConstraints tiling_constraints() const override { return {2, 6, 8}; }

    UmziIo tensors() const override {
        const auto dtype = uint32_t(fp16 ? UMZI_FLOAT16 : UMZI_FLOAT32);
        return {sizeof(UmziIo), {"input", {4, {0, 3, 0, 0}}, dtype, 0}, {"output", {4, {0, 3, 0, 0}}, dtype, 0}};
    }

    UmziIo describe(const UmziShape& request) const override {
        const auto input = checked(request);
        auto io = tensors();
        io.input.shape = request;
        io.input.bytes = rgb_bytes(input);
        io.output.shape = Shape{input.n, input.h * scale, input.w * scale}.nchw();
        io.output.bytes = io.input.bytes * scale * scale;
        return io;
    }

    UmziMemoryInfo estimate_memory(const UmziShape& request) const override {
        const auto input = checked(request);
        const auto padded = input.padded();
        ContextScope scope(context);
        const auto vram = umzi::device_vram_info(device);
        const auto sizes = activation_sizes(padded);
        UmziMemoryInfo info{};
        info.struct_size = sizeof(info);
        info.buffer_bytes[0] = net.bytes;
        info.buffer_bytes[1] = std::accumulate(sizes.begin(), sizes.end(), uint64_t(0));
        if (shape == input) {
            info.buffer_bytes[2] = conv_bytes + fft_bytes;
        } else {
            int dims[] = {padded.h, padded.w};
            size_t r2c_bytes = 0, c2r_bytes = 0;
            FFT_CHECK(fft->estimate_many(2, dims, nullptr, 1, 0, nullptr, 1, 0, r2c, padded.n * fourier, &r2c_bytes));
            FFT_CHECK(fft->estimate_many(2, dims, nullptr, 1, 0, nullptr, 1, 0, c2r, padded.n * fourier, &c2r_bytes));
            info.buffer_bytes[2] = std::min<uint64_t>(vram.free_bytes / 2, 256u << 20) + std::max(r2c_bytes, c2r_bytes);
        }
        info.buffer_bytes[3] = graph ? rgb_bytes(input) * (1 + scale * scale) : 0;
        info.device_free = vram.free_bytes;
        info.device_total = vram.total_bytes;
        return info;
    }

    void prepare_shape(const UmziShape& request) override {
        ContextScope scope(context);
        prepare(checked(request));
    }

    void run_host(const UmziShape& request, const void* input, void* output) override {
        ContextScope scope(context);
        prepare(checked(request));
        wait();
        ensure_staging();
        const auto bytes = rgb_bytes(shape);
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
            // Order this batch after the previous one on the device instead of blocking the host.
            if (pending) CHECK(cuStreamWaitEvent(target, complete, CU_EVENT_WAIT_DEFAULT));
            execute(input, output, target);
            pending = true; // Before the record, so teardown waits even if recording fails.
            CHECK(cuEventRecord(complete, target));
        } catch (...) {
            if (const auto failure = umzi::drain_failure(target); !failure.empty()) {
                pending = true; // Keep teardown waiting for work that may still be running.
                warn_option(failure);
            }
            throw;
        }
    }

    const char* kernel_message() const override {
        if (fp32_nchw) return "FiGSR FP32 with NCHW expansion and projection and FP32 cuFFT";
        if (fp32) return "FiGSR NHWC FP32 with FP32 cuFFT";
        return fp16 ? "FiGSR NHWC FP16 with FP32 cuFFT" : "FiGSR NHWC mixed precision with FP32 cuFFT";
    }

    uint32_t diagnostic_count() const override { return tune_convolutions ? 4 : 2; }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "fp32_accumulation_fallbacks", fp32_fallbacks};
        if (index == 1) return {sizeof(UmziDiagnostic), "cufft_workspace_bytes", int64_t(fft_bytes)};
        if (index == 2 && tune_convolutions) return {sizeof(UmziDiagnostic), "timed_convolution_candidates", timed_candidates};
        if (index == 3 && tune_convolutions) return {sizeof(UmziDiagnostic), "changed_convolution_plans", changed_plans};
        throw std::runtime_error("Invalid FiGSR diagnostic index");
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo info{};
        info.struct_size = sizeof(info);
        info.buffer_bytes[0] = net.bytes;
        if (shape.n) {
            info.prepared_input = shape.padded().nchw();
            info.buffer_bytes[1] = activation_bytes;
            info.buffer_bytes[2] = conv_bytes + fft_bytes;
            info.buffer_bytes[3] = stage_in ? rgb_bytes(shape) * (1 + scale * scale) : 0;
            std::copy(setup_ms.begin(), setup_ms.end(), info.shape_setup_ms);
            std::copy(device_used.begin(), device_used.end(), info.startup_device_used_bytes);
        }
        return info;
    }
};
}

std::unique_ptr<umzi::KernelSession> umzi::make_figsr(const ModelPackage& package, const UmziSessionOptions& options) {
    auto session = std::make_unique<Session>();
    session->open(package, options);
    return session;
}
