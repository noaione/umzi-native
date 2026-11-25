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
constexpr uint32_t fused_plan_schema = 1;
constexpr size_t fused_plan_file_limit = 256 * 1024;
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
    size_t elements() const { return size_t(cin) * cout * k * k; }
};

struct Network {
    Filter rgb{8, 16, 1}, first_skip{8, features, 1};
    std::array<std::array<Filter, 3>, 6> blocks{};
    Filter end{features, features, 3}, tail{64, 16, 3};
    size_t bytes = 0;

    explicit Network(size_t element_bytes = 2) {
        pack(rgb, element_bytes);
        pack(first_skip, element_bytes);
        for (size_t i = 0; i < blocks.size(); ++i) {
            auto& block = blocks[i];
            block = {{{i ? features : 8, features, 3}, {features, features, 3}, {features, 2 * features, 3}}};
            for (auto& filter : block) pack(filter, element_bytes);
        }
        pack(end, element_bytes);
        pack(tail, element_bytes);
    }

    void pack(Filter& f, size_t element_bytes) {
        f.weight = (bytes + 255) & ~size_t(255);
        const auto end = f.weight + f.elements() * element_bytes;
        f.bias = (end + 255) & ~size_t(255);
        bytes = f.bias + f.cout * sizeof(float);
    }

    std::vector<char> device_weights(const Network& source, std::string_view data, bool fp32) const {
        std::vector<char> result(bytes);
        auto convert = [&](const Filter& from, const Filter& to) {
            if (fp32) {
                const int taps = to.k * to.k;
                for (int out = 0; out < to.cout; ++out)
                    for (int in = 0; in < to.cin; ++in)
                        for (int tap = 0; tap < taps; ++tap) {
                            const auto src = (out * taps + tap) * to.cin + in;
                            const auto dst = (out * to.cin + in) * taps + tap;
                            std::memcpy(result.data() + to.weight + dst * sizeof(float),
                                        data.data() + from.weight + src * sizeof(float), sizeof(float));
                        }
            } else {
                for (size_t i = 0; i < to.elements(); ++i) {
                    float value;
                    std::memcpy(&value, data.data() + from.weight + i * sizeof(float), sizeof(float));
                    const auto half = umzi::float_to_half(value);
                    if ((half & 0x7fff) >= 0x7c00) throw std::runtime_error("SMoSR weights exceed the finite FP16 range");
                    std::memcpy(result.data() + to.weight + i * sizeof(half), &half, sizeof(half));
                }
            }
            std::memcpy(result.data() + to.bias, data.data() + from.bias, to.cout * sizeof(float));
        };
        convert(source.rgb, rgb);
        convert(source.first_skip, first_skip);
        for (size_t i = 0; i < blocks.size(); ++i)
            for (size_t j = 0; j < blocks[i].size(); ++j) convert(source.blocks[i][j], blocks[i][j]);
        convert(source.end, end);
        convert(source.tail, tail);
        return result;
    }
};

enum Kernel { PackF32, PackF16, InitActivations, Silu, Gate, Concat, ShuffleF32, ShuffleF16,
              PackFull, SiluFull, GateFull, ConcatFull, ShuffleFull, KernelCount };
constexpr std::array<const char*, KernelCount> names{
    "pack_f32", "pack_f16", "init_activations", "silu", "gated_residual", "concat", "shuffle_f32", "shuffle_f16",
    "pack_full", "silu_full", "gate_full", "concat_full", "shuffle_full"};
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
    bool fp16 = false, fp32 = false, graph = false, pending = false;
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
    bool tune_convolutions = false, fp16_accumulation = false, fuse_silu = false;
    std::filesystem::path cache_dir;
    umzi::Digest cache_environment{};
    UmziCacheStatus cache_status = UMZI_CACHE_DISABLED;
    std::string cache_diagnostic;
    int64_t timed_candidates = 0;
    std::array<double, 4> setup_ms{};
    std::array<uint64_t, 4> device_used{};
    std::map<std::array<int, 4>, umzi::CudnnBackendPlan> plans;
    std::map<std::array<CUdeviceptr, 3>, std::unique_ptr<umzi::CudnnDescriptor>> variants;

    size_t rgb_bytes(Shape input) const { return size_t(input.pixels()) * 3 * (fp16 ? 2 : 4); }
    size_t persistent_bytes() const {
        return net.bytes + (fp16 ? table_bytes : 0)
            + (fuse_silu && fp16 ? net.blocks.size() * 2 * features * sizeof(uint16_t) : 0);
    }
    CUdeviceptr fused_bias(size_t block, size_t convolution) const {
        if (!fuse_silu) return 0;
        if (!fp16) return weights + net.blocks[block][convolution].bias;
        return weights + net.bytes + table_bytes + (block * 2 + convolution) * features * sizeof(uint16_t);
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

    void cache_issue(UmziCacheStatus status, const std::exception& error) {
        cache_status = status;
        cache_diagnostic = error.what();
    }

    std::unique_ptr<umzi::CudnnDescriptor> load_fused_plan(const std::filesystem::path& path,
        const umzi::Digest& digest, const Filter& f, CUdeviceptr bias) {
        try {
            if (!std::filesystem::exists(path)) {
                if (cache_status == UMZI_CACHE_HIT) cache_status = UMZI_CACHE_MISS;
                return {};
            }
            auto payload = umzi::read_plan(path, umzi::smosr_id, fused_plan_schema, digest, fused_plan_file_limit);
            if (payload.empty() || std::find(payload.begin(), payload.end(), '\0') != payload.end())
                throw std::runtime_error("Invalid cached SMoSR plan JSON");
            payload.push_back('\0');
            umzi::CudnnBackendPlan restored;
            restored.plan = std::make_unique<umzi::CudnnDescriptor>(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
            restored.plan->set(CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, &cudnn);
            restored.plan->set(CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR,
                               payload.data(), int64_t(payload.size()));
            restored.plan->finalize();
            int64_t bytes = 0, count = 0;
            CHECK(cudnnBackendGetAttribute(restored.plan->get(), CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                           CUDNN_TYPE_INT64, 1, &count, &bytes));
            if (count != 1 || bytes != 0) throw std::runtime_error("Cached SMoSR plan requires workspace");
            CUdeviceptr scratch = 0;
            size_t scratch_bytes = 0;
            const float elapsed = umzi::time_cudnn_backend_plan(cudnn, stream, timing, restored,
                buffer[Activated], weights + f.weight, buffer[Projection], scratch, scratch_bytes, 0,
                nullptr, nullptr, bias);
            if (elapsed == std::numeric_limits<float>::infinity())
                throw std::runtime_error("Cached SMoSR plan did not execute");
            return std::move(restored.plan);
        } catch (const std::filesystem::filesystem_error& error) {
            cache_issue(UMZI_CACHE_UNAVAILABLE, error);
        } catch (const std::exception& error) {
            cache_issue(UMZI_CACHE_REBUILT, error);
        }
        return {};
    }

    void prepare_plan(const Filter& f, Shape padded, size_t limit, CUdeviceptr bias = 0) {
        const std::array key{f.cin, f.cout, f.k, int(bias != 0)};
        if (plans.find(key) != plans.end()) return;
        umzi::CudnnBackendPlan plan;
        std::filesystem::path cache_path;
        umzi::Digest cache_key{};
        if (bias && !cache_dir.empty() && cache_status != UMZI_CACHE_UNAVAILABLE) {
            try {
                umzi::Sha256 hash;
                hash.add(cache_environment);
                hash.add(std::array{padded.n, padded.h, padded.w, f.cin, f.cout, f.k});
                if (!fp16) hash.add(std::array{int(fp16_accumulation)});
                cache_key = hash.finish();
                cache_path = umzi::plan_path(cache_dir, "smosr-silu", cache_key);
                plan.plan = load_fused_plan(cache_path, cache_key, f, bias);
            } catch (const std::exception& error) {
                cache_issue(UMZI_CACHE_UNAVAILABLE, error);
            }
            if (plan.plan) {
                plans.emplace(key, std::move(plan));
                return;
            }
        }
        const bool body_accumulates_in_fp16 = fp16_accumulation && f.cin == features && f.k == 3;
        plan.graph = std::make_unique<umzi::CudnnConvGraph>(
            cudnn, padded.n, padded.h, padded.w, f.cin, f.cout, f.k, fp32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF,
            fp16 || body_accumulates_in_fp16,
            fp32 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC, bias != 0, !fp16);
        const bool tune = bias || (tune_convolutions && f.cin == features && f.k == 3);
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
                    buffer[Activated], weights + f.weight, buffer[Projection], scratch, scratch_bytes, fastest,
                    nullptr, nullptr, bias);
                candidate.plan = std::move(plan.plan);
                ++timed_candidates;
                if (elapsed < fastest) {
                    fastest = elapsed;
                    selected = i;
                }
            }
        }
        if (selected >= candidates.size() && bias) return;
        if (selected >= candidates.size())
            throw std::runtime_error("No cuDNN SMoSR plan for " + std::to_string(f.cin) + " to "
                                     + std::to_string(f.cout) + " channels, kernel " + std::to_string(f.k));
        plan.plan = std::move(candidates[selected].plan);
        plan.workspace = size_t(candidates[selected].workspace);
        if (!cache_path.empty() && cache_status != UMZI_CACHE_UNAVAILABLE) {
            try {
                const auto json = umzi::CudnnConvGraph::plan_json(*plan.plan);
                if (json.empty()) throw std::runtime_error("Could not serialize SMoSR fused plan");
                umzi::write_plan(cache_path, umzi::smosr_id, fused_plan_schema, cache_key,
                                 {json.begin(), json.end()}, fused_plan_file_limit);
            } catch (const std::exception& error) {
                cache_issue(UMZI_CACHE_UNAVAILABLE, error);
            }
        }
        workspace_bytes = std::max(workspace_bytes, plan.workspace);
        plans.emplace(key, std::move(plan));
    }

    template<class... Args>
    void launch(CUstream target, Kernel kernel, int count, Args... args) {
        const unsigned threads = fp16 && (kernel == Silu || kernel == Gate) ? 128 : 256;
        void* parameters[] = {&args...};
        CHECK(cuLaunchKernel(functions[kernel], (unsigned(count) + threads - 1) / threads, 1, 1,
                             threads, 1, 1, 0, target, parameters, nullptr));
    }

    bool conv(CUdeviceptr out, CUdeviceptr in, const Filter& f, CUdeviceptr bias = 0) {
        const auto fused = bias ? plans.find({f.cin, f.cout, f.k, 1}) : plans.end();
        const bool activated = fused != plans.end();
        const auto& plan = activated ? fused->second : plans.at({f.cin, f.cout, f.k, 0});
        const std::array key{in, weights + f.weight, out};
        auto variant = variants.find(key);
        if (variant == variants.end()) {
            auto descriptor = umzi::bind_conv_variant(reinterpret_cast<void*>(in),
                reinterpret_cast<void*>(weights + f.weight), reinterpret_cast<void*>(out),
                reinterpret_cast<void*>(workspace), activated ? reinterpret_cast<void*>(bias) : nullptr);
            variant = variants.emplace(key, std::move(descriptor)).first;
        }
        CHECK(cudnnBackendExecute(cudnn, plan.plan->get(), variant->second->get()));
        return activated;
    }

    void forward(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const auto padded = shape.padded();
        const int area = padded.h * padded.w, pixels = padded.pixels(), elements = pixels * features;
        const int pairs = elements / 2;
        const CUdeviceptr table = fp16 ? weights + net.bytes : 0;
        auto activate = [&](CUdeviceptr out, CUdeviceptr bias) {
            if (fp32) launch(target, SiluFull, elements, out, bias, area, elements);
            else launch(target, Silu, pairs, out, bias, table, pairs);
        };
        auto state = buffer[State], saved = buffer[Saved];
        launch(target, fp32 ? PackFull : fp16 ? PackF16 : PackF32,
               pixels * 8, buffer[RgbInput], input, shape.h, shape.w, pixels * 8);
        conv(buffer[RgbSkip], buffer[RgbInput], net.rgb);
        conv(saved, buffer[RgbInput], net.first_skip);
        for (size_t i = 0; i < net.blocks.size(); ++i) {
            const bool starts_middle = i == configuration[5];
            const bool ends_middle = i + 1 == configuration[5] + configuration[6];
            const auto& block = net.blocks[i];
            if (!conv(buffer[Hidden], i ? state : buffer[RgbInput], block[0], fused_bias(i, 0)))
                activate(buffer[Hidden], weights + block[0].bias);
            if (!conv(buffer[Activated], buffer[Hidden], block[1], fused_bias(i, 1)))
                activate(buffer[Activated], weights + block[1].bias);
            conv(buffer[Projection], buffer[Activated], block[2]);
            const auto out = starts_middle ? saved : state, shortcut = i ? state : saved;
            const auto bias = weights + block[2].bias, shortcut_bias = i ? 0 : weights + net.first_skip.bias;
            const auto middle = ends_middle ? saved : 0;
            if (fp32) launch(target, GateFull, elements, out, buffer[Projection], shortcut, bias, shortcut_bias,
                             middle, area, elements);
            else launch(target, Gate, pairs, out, buffer[Projection], shortcut, bias, shortcut_bias,
                        middle, table, pairs);
            if (starts_middle) std::swap(state, saved);
        }
        conv(buffer[Hidden], state, net.end);
        if (fp32) launch(target, ConcatFull, pixels * 64, buffer[Joined], buffer[RgbSkip], weights + net.rgb.bias,
                         buffer[Hidden], weights + net.end.bias, area, pixels * 64);
        else launch(target, Concat, pixels * 32, buffer[Joined], buffer[RgbSkip], weights + net.rgb.bias,
                    buffer[Hidden], weights + net.end.bias, pixels * 32, fp16);
        conv(buffer[RgbSkip], buffer[Joined], net.tail);
        const int count = shape.pixels() * 3 * scale * scale;
        launch(target, fp32 ? ShuffleFull : fp16 ? ShuffleF16 : ShuffleF32,
               count, output, buffer[RgbSkip], weights + net.tail.bias,
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
            const auto lane_bytes = size_t(padded.pixels()) * (fp32 ? 4 : 2);
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
            if (!cache_dir.empty() && cache_status != UMZI_CACHE_UNAVAILABLE) {
                cache_status = UMZI_CACHE_HIT;
                cache_diagnostic.clear();
            }
            if (tune_convolutions || fuse_silu)
                CHECK(cuMemsetD16Async(buffer[Activated], 0x3400, size_t(padded.pixels()) * features, stream));
            prepare_plan(net.rgb, padded, limit);
            prepare_plan(net.first_skip, padded, limit);
            for (const auto& block : net.blocks)
                for (const auto& filter : block) prepare_plan(filter, padded, limit);
            if (fuse_silu) {
                prepare_plan(net.blocks[0][0], padded, 0, fused_bias(0, 0));
                prepare_plan(net.blocks[0][1], padded, 0, fused_bias(0, 1));
            }
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
        if (package.schema != 1 && package.schema != 2) throw std::runtime_error("Unsupported SMoSR model payload version");
        fp16 = options.precision == UMZI_FP16;
        fp32 = options.precision == UMZI_FP32;
        if (fp32 && package.schema == 1)
            throw std::runtime_error("SMoSR FP32 requires a schema 2 package. Export the checkpoint again.");
        const Network source(package.schema == 1 ? 2 : 4);
        net = Network(fp32 ? 4 : 2);
        umzi::BinaryReader architecture(package.section(1));
        for (auto value : configuration)
            if (architecture.number<uint32_t>() != value) throw std::runtime_error("Unsupported SMoSR architecture");
        const auto data = package.section(3);
        if (architecture.size() || !package.section(2).empty() || package.section(4).size() != 32
            || data.size() != source.bytes)
            throw std::runtime_error("Invalid SMoSR metadata or weights");
        bool tune_set = false, accumulation_set = false, fusion_set = false;
        for (uint32_t i = 0; i < options.kernel_option_count; ++i) {
            const auto& option = options.kernel_options[i];
            const auto name = std::string_view(option.name ? option.name : "");
            if (option.value != 0 && option.value != 1)
                throw std::runtime_error("SMoSR kernel options require a value of 0 or 1");
            if (name == "tune_convolutions" && !tune_set) {
                tune_convolutions = option.value != 0;
                tune_set = true;
            } else if (name == "fp16_accumulation" && !accumulation_set) {
                fp16_accumulation = option.value != 0;
                accumulation_set = true;
            } else if (name == "fuse_silu" && !fusion_set) {
                fuse_silu = option.value != 0;
                fusion_set = true;
            } else {
                throw std::runtime_error("Invalid or duplicate SMoSR kernel option: " + std::string(name));
            }
        }
        if (fp32 && tune_convolutions) throw std::runtime_error("SMoSR tune_convolutions requires mixed or FP16 precision");
        if ((fp16 || fp32) && fp16_accumulation)
            throw std::runtime_error("SMoSR fp16_accumulation requires mixed precision");
        if (fuse_silu && fp32) throw std::runtime_error("SMoSR fuse_silu requires mixed or FP16 precision");
        graph = (options.flags & UMZI_CUDA_GRAPH) != 0;
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        ContextScope scope(context);
        const auto& image = umzi::load_kernel_module(device, module, smosr_images::embedded_cubins,
                                                     smosr_images::embedded_ptx);
        if (fuse_silu && options.cache_dir && *options.cache_dir) {
            try {
                cache_dir = std::filesystem::absolute(std::filesystem::u8path(options.cache_dir));
                std::filesystem::create_directories(cache_dir);
                umzi::Sha256 hash;
                umzi::hash_plan_environment(hash, umzi::smosr_id, fused_plan_schema, package.digest, image, device);
                cache_environment = hash.finish();
                cache_status = UMZI_CACHE_MISS;
            } catch (const std::exception& error) {
                cache_issue(UMZI_CACHE_UNAVAILABLE, error);
            }
        }
        for (size_t i = 0; i < names.size(); ++i)
            CHECK(cuModuleGetFunction(&functions[i], module, names[i]));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        if (tune_convolutions || fuse_silu)
            for (auto& event : timing) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        CHECK(cudnnCreate(&cudnn));
        CHECK(cuMemAlloc(&weights, persistent_bytes()));
        if (package.schema == 2) {
            const auto packed = net.device_weights(source, data, fp32);
            CHECK(cuMemcpyHtoD(weights, packed.data(), packed.size()));
        } else {
            CHECK(cuMemcpyHtoD(weights, data.data(), data.size()));
        }
        if (fuse_silu && fp16) {
            std::vector<uint16_t> biases;
            biases.reserve(source.blocks.size() * 2 * features);
            for (const auto& block : source.blocks) {
                for (size_t j = 0; j < 2; ++j) {
                    for (int i = 0; i < features; ++i) {
                        float value;
                        std::memcpy(&value, data.data() + block[j].bias + i * sizeof(value), sizeof(value));
                        biases.push_back(umzi::float_to_half(value));
                    }
                }
            }
            CHECK(cuMemcpyHtoD(fused_bias(0, 0), biases.data(), biases.size() * sizeof(uint16_t)));
        }
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
    umzi::TilingConstraints tiling_constraints() const override { return {1, 3, 4}; }

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
        if (fuse_silu && fp16_accumulation)
            return "SMoSR NHWC mixed precision with FP16 body accumulation and convolution and SiLU fusion";
        if (fuse_silu && !fp16) return "SMoSR NHWC mixed precision with convolution and SiLU fusion";
        if (fp16_accumulation) return "SMoSR NHWC mixed precision with FP16 accumulation in body convolutions";
        if (fuse_silu) return "SMoSR NHWC FP16 with convolution and SiLU fusion";
        return fp32 ? "SMoSR NCHW FP32" : fp16 ? "SMoSR NHWC FP16" : "SMoSR NHWC mixed precision";
    }

    uint32_t diagnostic_count() const override { return fuse_silu ? 2 : 1; }
    const char* cache_message() const override { return cache_diagnostic.c_str(); }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "timed_candidates", timed_candidates};
        if (index == 1 && fuse_silu) {
            const auto count = std::count_if(plans.begin(), plans.end(), [](const auto& plan) { return plan.first[3]; });
            return {sizeof(UmziDiagnostic), "fused_silu_shapes", int64_t(count)};
        }
        throw std::runtime_error("Invalid SMoSR diagnostic index");
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo info{};
        info.struct_size = sizeof(info);
        info.cache_status = cache_status;
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
