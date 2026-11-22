#define UMZI_BUILD
#include "../include/umzi_native.h"

#include <charconv>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;

uint32_t parse_u32(const std::string& value, const std::string& name) {
    uint32_t number{};
    const auto* end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, number);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::runtime_error("Invalid number for " + name);
    return number;
}

int64_t parse_i64(const std::string& value, const std::string& name) {
    int64_t number{};
    const auto* end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, number);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::runtime_error("Invalid integer value for " + name);
    return number;
}

void print_usage(std::ostream& out) {
    out << "Usage: umzi_native run --model file --batch N --height H --width W --input raw --output raw\n"
        << "  --precision mixed|fp16|fp32   Select a precision supported by the model (default: mixed)\n"
        << "  --device index                Select a CUDA device (default: 0)\n"
        << "  --cache-dir path              Read and write tuned plans in this directory\n"
        << "  --direct                      Dispatch inference without capturing a CUDA Graph\n"
        << "  --option name=value           Pass an integer kernel option (repeatable)\n";
}

int run_cli(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(argc < 2 ? std::cerr : std::cout);
        return argc < 2 ? 2 : 0;
    }
    if (std::string(argv[1]) != "run") {
        std::cerr << "Unknown command: " << argv[1] << '\n';
        print_usage(std::cerr);
        return 2;
    }
    if (argc == 2 || (argc == 3 && (std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h"))) {
        print_usage(argc == 2 ? std::cerr : std::cout);
        return argc == 2 ? 2 : 0;
    }
    try {
        UmziSessionOptions options{sizeof(UmziSessionOptions), 0, UMZI_MIXED, UMZI_CUDA_GRAPH};
        std::vector<std::pair<std::string, int64_t>> parsed_options;
        std::string model, input, output;
        uint32_t batch = 0, height = 0, width = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string name = argv[i];
            if (name == "--direct") {
                options.flags &= ~UMZI_CUDA_GRAPH;
                continue;
            }
            if (++i >= argc) throw std::runtime_error("Missing value for " + name);
            if (name == "--model") model = argv[i];
            else if (name == "--input") input = argv[i];
            else if (name == "--output") output = argv[i];
            else if (name == "--cache-dir") options.cache_dir = argv[i];
            else if (name == "--batch") batch = parse_u32(argv[i], name);
            else if (name == "--height") height = parse_u32(argv[i], name);
            else if (name == "--width") width = parse_u32(argv[i], name);
            else if (name == "--device") {
                const auto device = parse_u32(argv[i], name);
                if (device > INT_MAX) throw std::runtime_error("Invalid CUDA device index");
                options.device = static_cast<int32_t>(device);
            } else if (name == "--precision") {
                const std::string mode = argv[i];
                if (mode == "fp16") options.precision = UMZI_FP16;
                else if (mode == "mixed") options.precision = UMZI_MIXED;
                else if (mode == "fp32") options.precision = UMZI_FP32;
                else throw std::runtime_error("Precision must be mixed, fp16, or fp32");
            } else if (name == "--option") {
                const std::string entry = argv[i];
                const auto eq = entry.find('=');
                if (eq == std::string::npos || eq == 0 || eq + 1 >= entry.size())
                    throw std::runtime_error("Kernel options must use name=value format: " + entry);
                parsed_options.emplace_back(entry.substr(0, eq), parse_i64(entry.substr(eq + 1), entry.substr(0, eq)));
            } else {
                throw std::runtime_error("Invalid option " + name);
            }
        }
        if (model.empty() || input.empty() || output.empty())
            throw std::runtime_error("--model, --input and --output are required");
        std::vector<UmziOption> kernel_options;
        kernel_options.reserve(parsed_options.size());
        for (const auto& [name, value] : parsed_options) kernel_options.push_back({name.c_str(), value});
        options.kernel_options = kernel_options.data();
        options.kernel_option_count = static_cast<uint32_t>(kernel_options.size());

        std::unique_ptr<UmziSession, decltype(&umzi_session_close)> session(
            umzi_session_open(model.c_str(), &options), umzi_session_close);
        if (!session) throw std::runtime_error(umzi_error());
        const UmziShape shape{4, {batch, 3, height, width}};
        UmziIo io{};
        io.struct_size = sizeof(io);
        if (umzi_session_describe(session.get(), &shape, &io)) throw std::runtime_error(umzi_error());
        if (fs::file_size(fs::u8path(input)) != io.input.bytes)
            throw std::runtime_error("Raw input must contain exactly " + std::to_string(io.input.bytes) + " bytes");
        std::vector<char> pixels(static_cast<size_t>(io.input.bytes));
        std::vector<char> restored(static_cast<size_t>(io.output.bytes));
        std::ifstream file(fs::u8path(input), std::ios::binary);
        if (!file.read(pixels.data(), pixels.size()))
            throw std::runtime_error("Could not read raw input");
        if (umzi_session_run(session.get(), &shape, pixels.data(), pixels.size(), restored.data(), restored.size()))
            throw std::runtime_error(umzi_error());
        std::ofstream out(fs::u8path(output), std::ios::binary);
        if (!out.write(restored.data(), restored.size()))
            throw std::runtime_error("Could not write restored RGB output");
        if (*umzi_session_cache_message(session.get()))
            std::cerr << "Plan cache: " << umzi_session_cache_message(session.get()) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** wide_argv) {
    std::vector<std::string> arguments;
    arguments.reserve(argc);
    for (int i = 0; i < argc; ++i) arguments.push_back(fs::path(wide_argv[i]).u8string());
    std::vector<char*> argv;
    argv.reserve(argc);
    for (auto& text : arguments) argv.push_back(text.data());
    return run_cli(argc, argv.data());
}
#else
int main(int argc, char** argv) {
    return run_cli(argc, argv);
}
#endif
