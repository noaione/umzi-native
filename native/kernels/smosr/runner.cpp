#include "../../runtime/binary.h"
#include "cudnn_conv_plan.h"
#include "smosr_images.h"

#include <climits>
#include <map>

namespace {
using umzi::ContextScope;
#define CHECK(call) umzi::check((call), #call)

constexpr int scale = 2, context_pixels = 2, features = 48;
constexpr int half_values = 1 << 16;
constexpr size_t table_bytes = 2 * half_values * sizeof(uint16_t);
constexpr std::array<uint32_t, 8> configuration{3, 3, scale, context_pixels, features, 2, 3, 1};

struct Shape {
    int n = 0, h = 0, w = 0;
    bool operator==(const Shape& other) const { return n == other.n && h == other.h && w == other.w; }
    int pixels() const { return n * h * w; }
    UmziShape nchw() const { return {4, {uint32_t(n), 3, uint32_t(h), uint32_t(w)}}; }
    Shape padded() const { return {n, h + 2 * context_pixels, w + 2 * context_pixels}; }
};

Shape checked(const UmziShape& input) {
    const auto* d = input.dimensions;
    const uint64_t n = d[0], h = d[2], w = d[3];
    if (input.rank != 4 || !n || d[1] != 3 || h <= context_pixels || w <= context_pixels
        || n > INT_MAX || h > INT_MAX / 2 || w > INT_MAX / 2
        || (h + 4) * (w + 4) > uint64_t(INT_MAX / (2 * features)) / n)
        throw std::runtime_error("SMoSR expects a supported NCHW RGB shape with height and width at least 3");
    return {int(n), int(h), int(w)};
}

struct Filter {
    int cin, cout, k;
    size_t weight = 0, bias = 0;
};

struct Network {
    Filter rgb{8, 16, 1}, first_skip{8, features, 1};
    std::array<std::array<Filter, 3>, 6> blocks{};
    Filter end{features, features, 3}, tail{64, 16, 3};
    size_t bytes = 0;

    Network() {
        pack(rgb);
        pack(first_skip);
        for (size_t i = 0; i < blocks.size(); ++i) {
            auto& block = blocks[i];
            block = {{{i ? features : 8, features, 3}, {features, features, 3}, {features, 2 * features, 3}}};
            for (auto& filter : block) pack(filter);
        }
        pack(end);
        pack(tail);
    }

    void pack(Filter& f) {
        f.weight = (bytes + 255) & ~size_t(255);
        const auto end = f.weight + size_t(f.cin) * f.cout * f.k * f.k * 2;
        f.bias = (end + 255) & ~size_t(255);
        bytes = f.bias + f.cout * sizeof(float);
    }
};

enum Kernel { PackF32, PackF16, InitActivations, Silu, Gate, Concat, ShuffleF32, ShuffleF16, KernelCount };
constexpr std::array<const char*, KernelCount> names{
    "pack_f32", "pack_f16", "init_activations", "silu", "gated_residual", "concat", "shuffle_f32", "shuffle_f16"};
enum Buffer { RgbInput, RgbSkip, Saved, State, Hidden, Activated, Projection, Joined, BufferCount };
// Eager and captured dispatch share one stream, so dead activations can overlap.
constexpr int rgb_offset = 0, saved_offset = 16, state_offset = saved_offset + features;
constexpr int activated_offset = state_offset + features, projection_offset = activated_offset + features;
constexpr int arena_channels = projection_offset + 2 * features;
constexpr std::array<int, BufferCount> buffer_offsets{
    activated_offset, rgb_offset, saved_offset, state_offset, projection_offset, activated_offset, projection_offset,
    saved_offset};

class Session final : public umzi::KernelSession {
    Network net;
    bool fp16 = false, graph = false, pending = false;
    CUdevice device{};
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUstream stream = nullptr;
    CUevent complete = nullptr;
    std::array<CUevent, 2> timing{};
    cudnnHandle_t cudnn = nullptr;
    std::array<CUfunction, KernelCount> functions{};
    CUgraphExec executable = nullptr;
    CUdeviceptr weights = 0, arena = 0, stage_in = 0, stage_out = 0, workspace = 0;
    std::array<CUdeviceptr, BufferCount> buffer{};
    Shape shape;
    size_t activation_bytes = 0, workspace_bytes = 0;
    bool tune_convolutions = false;
    int64_t timed_candidates = 0;
    std::array<double, 4> setup_ms{};
    std::array<uint64_t, 4> device_used{};
    std::map<std::array<int, 3>, umzi::CudnnBackendPlan> plans;
    std::map<std::array<CUdeviceptr, 3>, std::unique_ptr<umzi::CudnnDescriptor>> variants;

    size_t rgb_bytes(Shape input) const { return size_t(input.pixels()) * 3 * (fp16 ? 2 : 4); }
    size_t persistent_bytes() const { return net.bytes + (fp16 ? table_bytes : 0); }

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
        for (auto* pointer : {&arena, &stage_in, &stage_out, &workspace}) {
            if (*pointer) cuMemFree(*pointer);
            *pointer = 0;
        }
        buffer.fill(0);
        shape = {};
        activation_bytes = workspace_bytes = 0;
        timed_candidates = 0;
        setup_ms = {};
        device_used = {};
    }

    void prepare_plan(const Filter& f, Shape padded, size_t limit) {
        const std::array key{f.cin, f.cout, f.k};
        if (plans.find(key) != plans.end()) return;
        umzi::CudnnBackendPlan plan;
        plan.graph = std::make_unique<umzi::CudnnConvGraph>(
            cudnn, padded.n, padded.h, padded.w, f.cin, f.cout, f.k, CUDNN_DATA_HALF, fp16);
        const bool tune = tune_convolutions && f.cin == features && f.k == 3;
        auto candidates = plan.graph->candidates(cudnn, tune ? 0 : limit, !tune);
        size_t selected = 0;
        if (tune) {
            CUdeviceptr scratch = 0;
            size_t scratch_bytes = 0;
            float fastest = std::numeric_limits<float>::infinity();
            selected = candidates.size();
            for (size_t i = 0; i < candidates.size(); ++i) {
                auto& candidate = candidates[i];
                plan.plan = std::move(candidate.plan);
                plan.workspace = size_t(candidate.workspace);
                const float elapsed = umzi::time_cudnn_backend_plan(cudnn, stream, timing, plan,
                    buffer[Activated], weights + f.weight, buffer[Projection], scratch, scratch_bytes, fastest);
                candidate.plan = std::move(plan.plan);
                ++timed_candidates;
                if (elapsed < fastest) {
                    fastest = elapsed;
                    selected = i;
                }
            }
        }
        if (selected >= candidates.size())
            throw std::runtime_error("No cuDNN SMoSR plan for " + std::to_string(f.cin) + " to "
                                     + std::to_string(f.cout) + " channels, kernel " + std::to_string(f.k));
        plan.plan = std::move(candidates[selected].plan);
        plan.workspace = size_t(candidates[selected].workspace);
        workspace_bytes = std::max(workspace_bytes, plan.workspace);
        plans.emplace(key, std::move(plan));
    }

    template<class... Args>
    void launch(CUstream target, Kernel kernel, int count, Args... args) {
        constexpr unsigned threads = 256;
        void* parameters[] = {&args...};
        CHECK(cuLaunchKernel(functions[kernel], (unsigned(count) + threads - 1) / threads, 1, 1,
                             threads, 1, 1, 0, target, parameters, nullptr));
    }

    void conv(CUdeviceptr out, CUdeviceptr in, const Filter& f) {
        const auto& plan = plans.at({f.cin, f.cout, f.k});
        const std::array key{in, weights + f.weight, out};
        auto variant = variants.find(key);
        if (variant == variants.end()) {
            auto descriptor = umzi::bind_conv_variant(reinterpret_cast<void*>(in),
                reinterpret_cast<void*>(weights + f.weight), reinterpret_cast<void*>(out),
                reinterpret_cast<void*>(workspace));
            variant = variants.emplace(key, std::move(descriptor)).first;
        }
        CHECK(cudnnBackendExecute(cudnn, plan.plan->get(), variant->second->get()));
    }

    void forward(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const int pixels = shape.padded().pixels(), pairs = pixels * features / 2;
        const CUdeviceptr table = fp16 ? weights + net.bytes : 0;
        auto state = buffer[State], saved = buffer[Saved];
        launch(target, fp16 ? PackF16 : PackF32, pixels * 8, buffer[RgbInput], input, shape.h, shape.w, pixels * 8);
        conv(buffer[RgbSkip], buffer[RgbInput], net.rgb);
        conv(saved, buffer[RgbInput], net.first_skip);
        for (size_t i = 0; i < net.blocks.size(); ++i) {
            const bool starts_middle = i == configuration[5];
            const bool ends_middle = i + 1 == configuration[5] + configuration[6];
            const auto& block = net.blocks[i];
            conv(buffer[Hidden], i ? state : buffer[RgbInput], block[0]);
            launch(target, Silu, pairs, buffer[Hidden], weights + block[0].bias, table, pairs);
            conv(buffer[Activated], buffer[Hidden], block[1]);
            launch(target, Silu, pairs, buffer[Activated], weights + block[1].bias, table, pairs);
            conv(buffer[Projection], buffer[Activated], block[2]);
            launch(target, Gate, pairs, starts_middle ? saved : state, buffer[Projection], i ? state : saved,
                   weights + block[2].bias, i ? CUdeviceptr(0) : weights + net.first_skip.bias,
                   ends_middle ? saved : CUdeviceptr(0), table, pairs);
            if (starts_middle) std::swap(state, saved);
        }
        conv(buffer[Hidden], state, net.end);
        launch(target, Concat, pixels * 32, buffer[Joined], buffer[RgbSkip], weights + net.rgb.bias,
               buffer[Hidden], weights + net.end.bias, pixels * 32, fp16);
        conv(buffer[RgbSkip], buffer[Joined], net.tail);
        const int count = shape.pixels() * 3 * scale * scale;
        launch(target, fp16 ? ShuffleF16 : ShuffleF32, count, output, buffer[RgbSkip], weights + net.tail.bias,
               shape.h, shape.w, count);
    }

    void prepare(Shape requested) {
        wait();
        if (shape == requested) return;
        release_shape();
        try {
            const auto padded = requested.padded();
            auto start = umzi::Clock::now();
            device_used[0] = umzi::device_used_bytes();
            const auto lane_bytes = size_t(padded.pixels()) * 2;
            activation_bytes = lane_bytes * arena_channels;
            CHECK(cuMemAlloc(&arena, activation_bytes));
            for (size_t i = 0; i < buffer.size(); ++i)
                buffer[i] = arena + lane_bytes * buffer_offsets[i];
            const size_t input_bytes = rgb_bytes(requested);
            CHECK(cuMemAlloc(&stage_in, input_bytes));
            CHECK(cuMemAlloc(&stage_out, input_bytes * scale * scale));
            device_used[1] = umzi::device_used_bytes();
            setup_ms[0] = umzi::elapsed_ms(start);

            start = umzi::Clock::now();
            size_t available = 0, total = 0;
            CHECK(cuMemGetInfo(&available, &total));
            const auto limit = std::min<size_t>(available / 2, 256u << 20);
            CHECK(cudnnSetStream(cudnn, stream));
            if (tune_convolutions)
                CHECK(cuMemsetD16Async(buffer[Activated], 0x3400, size_t(padded.pixels()) * features, stream));
            prepare_plan(net.rgb, padded, limit);
            prepare_plan(net.first_skip, padded, limit);
            for (const auto& block : net.blocks)
                for (const auto& filter : block) prepare_plan(filter, padded, limit);
            prepare_plan(net.end, padded, limit);
            prepare_plan(net.tail, padded, limit);
            if (workspace_bytes) CHECK(cuMemAlloc(&workspace, workspace_bytes));
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
            CHECK(cudnnSetStream(cudnn, target));
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
        if (package.schema != 1) throw std::runtime_error("Unsupported SMoSR model payload version");
        umzi::BinaryReader architecture(package.section(1));
        for (auto value : configuration)
            if (architecture.number<uint32_t>() != value) throw std::runtime_error("Unsupported SMoSR architecture");
        const auto data = package.section(3);
        if (architecture.size() || !package.section(2).empty() || package.section(4).size() != 32
            || data.size() != net.bytes)
            throw std::runtime_error("Invalid SMoSR metadata or weights");
        if (options.precision != UMZI_MIXED && options.precision != UMZI_FP16)
            throw std::runtime_error("SMoSR supports mixed and FP16 precision");
        if (options.kernel_option_count) {
            const auto& option = options.kernel_options[0];
            if (options.kernel_option_count != 1 || !option.name || std::string_view(option.name) != "tune_convolutions"
                || (option.value != 0 && option.value != 1))
                throw std::runtime_error("Specify SMoSR tune_convolutions once with a value of 0 or 1");
            tune_convolutions = option.value != 0;
        }
        fp16 = options.precision == UMZI_FP16;
        graph = (options.flags & UMZI_CUDA_GRAPH) != 0;
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        ContextScope scope(context);
        umzi::load_kernel_module(device, module, smosr_images::embedded_cubins, smosr_images::embedded_ptx);
        for (size_t i = 0; i < names.size(); ++i)
            CHECK(cuModuleGetFunction(&functions[i], module, names[i]));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        if (tune_convolutions)
            for (auto& event : timing) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        CHECK(cudnnCreate(&cudnn));
        CHECK(cuMemAlloc(&weights, persistent_bytes()));
        CHECK(cuMemcpyHtoD(weights, data.data(), net.bytes));
        if (fp16) {
            launch(stream, InitActivations, half_values, weights + net.bytes);
            CHECK(cuStreamSynchronize(stream));
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
            for (auto event : timing) if (event) cuEventDestroy(event);
            if (stream) cuStreamDestroy(stream);
            if (module) cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        cuDevicePrimaryCtxRelease(device);
    }

    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "smosr", scale, 0}; }

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

    const char* kernel_message() const override { return fp16 ? "SMoSR NHWC FP16" : "SMoSR NHWC mixed precision"; }

    uint32_t diagnostic_count() const override { return 1; }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "timed_candidates", timed_candidates};
        throw std::runtime_error("Invalid SMoSR diagnostic index");
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo info{};
        info.struct_size = sizeof(info);
        info.cache_status = UMZI_CACHE_DISABLED;
        info.buffer_bytes[0] = persistent_bytes();
        if (shape.n) {
            info.prepared_input = shape.padded().nchw();
            info.buffer_bytes[1] = activation_bytes;
            info.buffer_bytes[2] = workspace_bytes;
            info.buffer_bytes[3] = rgb_bytes(shape) * (1 + scale * scale);
            std::copy(setup_ms.begin(), setup_ms.end(), info.shape_setup_ms);
            std::copy(device_used.begin(), device_used.end(), info.startup_device_used_bytes);
        }
        return info;
    }
};
}

std::unique_ptr<umzi::KernelSession> umzi::make_smosr(const ModelPackage& package, const UmziSessionOptions& options) {
    auto session = std::make_unique<Session>();
    session->open(package, options);
    return session;
}
