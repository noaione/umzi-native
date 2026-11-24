#include "../../runtime/binary.h"
#include "../../runtime/kernel.h"
#include "cudnn_conv_plan.h"
#include "cufft_api.h"
#include "figsr_images.h"

#include <climits>
#include <cmath>
#include <map>

namespace {
using umzi::ContextScope;
#define CHECK(call) umzi::check((call), #call)

constexpr int scale = 2, border = 4, features = 32, hidden = 80, fourier = 8, group_blocks = 12;
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
    std::unique_ptr<umzi::CufftApi> fft;
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
        for (auto& pointer : buffers) if (pointer) cuMemFree(pointer);
        buffers.fill(0);
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
            CHECK(cuMemAlloc(&fft_workspace, fft_bytes));
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
        size_t scratch_bytes = conv_bytes;
        const auto elapsed = [&](const umzi::CudnnBackendPlan& plan, float best) {
            ++timed_candidates;
            return umzi::time_cudnn_backend_plan(cudnn, stream, tuning_events, plan, buffers[input],
                weights + filter.weight, buffers[output], conv_workspace, scratch_bytes, best);
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
                       area, pixels, ortho, fp16, fp32);

            int offset = hidden - 3 * fourier;
            // Both Fourier staging buffers are dead after the inverse gate.
            for (const auto* filter : {&block.square, &block.row, &block.column}) {
                const int count = pixels * fourier;
                launch_body(target, SplitSpatial, SplitSpatialNchw, count, area, b[Spectral], b[Expanded], expand_bias,
                            offset, count);
                if (fuse_row_gate && filter == &block.row) {
                    const int threads = 32 * ((spatial.w + 15) / 16) * spatial.h * spatial.n;
                    launch(target, RowConvGate, threads, b[Values], b[Spectral], weights + filter->weight,
                           weights + filter->bias, b[Expanded], expand_bias, spatial.w, spatial.h * spatial.n, offset);
                } else {
                    conv(b[Fourier], b[Spectral], *filter, spatial);
                    launch_body(target, GateSpatial, GateSpatialNchw, count, area, b[Values], b[Fourier], weights + filter->bias,
                                b[Expanded], expand_bias, offset, count);
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
        wait();
        if (shape == requested) return;
        release_shape();
        try {
            const auto padded = requested.padded();
            const Shape frequency{padded.n, padded.h, padded.w / 2 + 1};
            const size_t image = size_t(padded.pixels()) * element_bytes();
            const size_t spectrum = size_t(frequency.pixels()) * element_bytes();
            const size_t fft_input = size_t(padded.pixels()) * fourier * sizeof(float);
            const size_t fft_output = size_t(frequency.pixels()) * fourier * 2 * sizeof(float);
            const std::array<size_t, BufferCount> bytes{
                features * image, features * image, features * image, features * image,
                2 * hidden * image, hidden * image, fft_input, fft_output,
                2 * fourier * spectrum, 2 * fourier * spectrum};
            auto start = umzi::Clock::now();
            device_used[0] = umzi::device_used_bytes();
            for (size_t i = 0; i < bytes.size(); ++i) {
                CHECK(cuMemAlloc(&buffers[i], bytes[i]));
                activation_bytes += bytes[i];
            }
            const size_t input_bytes = rgb_bytes(requested);
            CHECK(cuMemAlloc(&stage_in, input_bytes));
            CHECK(cuMemAlloc(&stage_out, input_bytes * scale * scale));
            device_used[1] = umzi::device_used_bytes();
            setup_ms[0] = umzi::elapsed_ms(start);

            start = umzi::Clock::now();
            prepare_fft(padded);
            bind_stream(stream);
            size_t available = 0, total = 0;
            CHECK(cuMemGetInfo(&available, &total));
            const auto cap = std::min<size_t>(available / 2, 256u << 20);
            for (const auto* filter : {&net.head, &net.end, &net.combine, &net.tail}) prepare_conv(padded, *filter, cap);
            const auto& block = net.blocks.front();
            for (const auto* filter : {&block.expand, &block.square, &block.row, &block.column, &block.project})
                if ((!fuse_row_gate || filter != &block.row) && (!fuse_expand_gate || filter != &block.expand))
                    prepare_conv(padded, *filter, cap);
            prepare_conv(frequency, block.fpe, cap);
            prepare_conv(frequency, block.fdc, cap);
            if (conv_bytes) CHECK(cuMemAlloc(&conv_workspace, conv_bytes));
            if (tune_convolutions) {
                if (!fuse_expand_gate) tune_conv(padded, block.expand, Normalized, Expanded);
                tune_conv(padded, block.project, Values, Normalized);
                for (const auto* filter : {&block.square, &block.row, &block.column})
                    if (!fuse_row_gate || filter != &block.row) tune_conv(padded, *filter, Spectral, Fourier);
            }
            device_used[2] = umzi::device_used_bytes();
            setup_ms[1] = umzi::elapsed_ms(start);

            start = umzi::Clock::now();
            shape = requested;
            CHECK(cuMemsetD8Async(stage_in, 0, input_bytes, stream));
            forward(stage_in, stage_out, stream);
            CHECK(cuStreamSynchronize(stream));
            setup_ms[2] = umzi::elapsed_ms(start);
            if (graph) {
                start = umzi::Clock::now();
                umzi::capture_cuda_graph(stream, executable, [&] { forward(stage_in, stage_out, stream); });
                setup_ms[3] = umzi::elapsed_ms(start);
            }
            device_used[3] = umzi::device_used_bytes();
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
        if (!fp32 && fp32_nchw) throw std::runtime_error("FiGSR fp32_nchw requires FP32 precision");
        if ((fp16 || fp32) && fp16_expand_accumulation)
            throw std::runtime_error("FiGSR fp16_expand_accumulation requires mixed precision");
        if ((fp16 || fp32) && fuse_row_gate) throw std::runtime_error("FiGSR fuse_row_gate requires mixed precision");
        if (fp32 && fuse_expand_gate) throw std::runtime_error("FiGSR fuse_expand_gate requires mixed or FP16 precision");
        if (!fp32) net = Network(sizeof(uint16_t));
        graph = (options.flags & UMZI_CUDA_GRAPH) != 0;
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        ContextScope scope(context);
        fft = std::make_unique<umzi::CufftApi>();
        umzi::load_kernel_module(device, module, figsr_images::embedded_cubins, figsr_images::embedded_ptx);
        for (size_t i = 0; i < names.size(); ++i) CHECK(cuModuleGetFunction(&functions[i], module, names[i]));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        if (tune_convolutions)
            for (auto& event : tuning_events) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        CHECK(cudnnCreate(&cudnn));
        CHECK(cuMemAlloc(&weights, net.bytes));
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
        if (!context) return;
        if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            if (pending) cuEventSynchronize(complete);
            if (stream) cuStreamSynchronize(stream);
            release_shape();
            if (weights) cuMemFree(weights);
            if (cudnn) cudnnDestroy(cudnn);
            if (complete) cuEventDestroy(complete);
            for (auto event : tuning_events) if (event) cuEventDestroy(event);
            if (stream) cuStreamDestroy(stream);
            if (module) cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        cuDevicePrimaryCtxRelease(device);
    }

    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "figsr", scale, 0}; }

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

    void prepare_shape(const UmziShape& request) override {
        ContextScope scope(context);
        prepare(checked(request));
    }

    void run_host(const UmziShape& request, const void* input, void* output) override {
        ContextScope scope(context);
        prepare(checked(request));
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
            execute(input, output, target);
            CHECK(cuEventRecord(complete, target));
            pending = true;
        } catch (...) {
            cuStreamSynchronize(target);
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
            info.buffer_bytes[3] = rgb_bytes(shape) * (1 + scale * scale);
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
