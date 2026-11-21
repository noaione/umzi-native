#pragma once

#include <stdint.h>

#ifdef _WIN32
#  ifdef GATERV3_BUILD
#    define GT_API __declspec(dllexport)
#  else
#    define GT_API __declspec(dllimport)
#  endif
#else
#  define GT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GtRunner GtRunner;
typedef struct CUstream_st* GtStream;

typedef enum GtCreateFlags {
    GT_CREATE_GRAPH = 1,
    GT_CREATE_FP16 = 2,
    GT_CREATE_FUSE_SILU = 4,        // Try cuDNN convolution/SiLU fusion.
    GT_CREATE_TUNE_PROJECTIONS = 8, // Time more cuBLASLt projection candidates under CUDA Graph replay.
    GT_CREATE_FUSE_GATE_POOL = 16,  // Fuse eligible 128x128 grouped gate/pooling sites.
    GT_CREATE_FP16_QKV = 32,        // Use FP16 QKV accumulation in a mixed-precision session.
} GtCreateFlags;

// Indices in GtSessionInfo.optimization_stats and gt_optimization_stats.
typedef enum GtOptimizationStat {
    GT_OPT_REPLAY_TUNED_SHAPES,
    GT_OPT_FUSED_GATE_POOL_SITES,
    GT_OPT_FP16_QKV_PROJECTIONS,
    GT_OPT_SPLIT_GATE_REGISTERS,
    GT_OPT_FUSED_GATE_REGISTERS,
    GT_OPT_STAT_COUNT,
} GtOptimizationStat;

typedef enum GtCacheStatus {
    GT_CACHE_DISABLED,
    GT_CACHE_MISS,
    GT_CACHE_HIT,
    GT_CACHE_REBUILT,
    GT_CACHE_UNAVAILABLE,
} GtCacheStatus;

typedef enum GtStatus {
    GT_STATUS_OK = 0,
    GT_STATUS_ERROR = -1,
    GT_STATUS_OUT_OF_MEMORY = -2,
} GtStatus;

// Uses CUDA device 0. The bundle contains schedule.bin, weights.bin and kernels.cubin.
GT_API GtRunner* gt_create(const char* utf8_directory, int batch, int graph);
GT_API GtRunner* gt_create_fp16(const char* utf8_directory, int batch, int graph);
// An empty or null cache directory disables disk caching.
GT_API GtRunner* gt_create_ex(const char* utf8_directory, int batch, uint32_t flags,
                             const char* utf8_cache_directory);
GT_API void gt_destroy(GtRunner* runner);
GT_API const char* gt_error(void);
GT_API GtCacheStatus gt_cache_status(const GtRunner* runner);
GT_API const char* gt_cache_message(const GtRunner* runner);
// Model loading, plan lookup/tuning, allocation/warmup and graph capture, in ms.
GT_API void gt_setup_timings(const GtRunner* runner, double milliseconds[4]);
GT_API int gt_fused_silu_count(const GtRunner* runner);
GT_API const char* gt_fusion_message(const GtRunner* runner);
// Replay-tuned shapes on this load, fused gate/pool sites, selective FP16
// projections, and split/fused gate registers per thread.
GT_API void gt_optimization_stats(const GtRunner* runner, int values[GT_OPT_STAT_COUNT]);

// Synchronous host I/O. Both buffers contain batch × 3 × 128 × 128 FP32 values in NCHW order.
GT_API int gt_run(GtRunner* runner, const float* input, float* output);
// The FP16 entry point uses IEEE binary16 bit patterns in the same RGB layout.
GT_API int gt_run_fp16(GtRunner* runner, const uint16_t* input, uint16_t* output);
// Weights, activations, workspace and RGB input/output, respectively, in bytes.
GT_API void gt_buffer_sizes(const GtRunner* runner, uint64_t bytes[4]);
// Device-wide usage sampled at context setup, workspace allocation, after
// tuning, and after warmup/capture, respectively, in bytes.
GT_API void gt_startup_memory(const GtRunner* runner, uint64_t bytes[4]);

// The buffers use the runner's RGB precision. Populate input before enqueue and synchronize the
// selected stream before reading output. Use the device's primary context. Serialize
// calls on each runner, including any outstanding work, before using another stream
// or destroying the runner.
GT_API uint64_t gt_input(const GtRunner* runner);
GT_API uint64_t gt_output(const GtRunner* runner);
GT_API int gt_enqueue(GtRunner* runner, GtStream stream);

// Profile into a zero-initialized array with one float per operation.
GT_API int gt_operation_count(const GtRunner* runner);
GT_API int gt_profile(GtRunner* runner, float* milliseconds, int count);

#ifdef __cplusplus
}
#endif

#undef GT_API
