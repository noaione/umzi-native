#pragma once

#include <stdint.h>

#ifdef GATERV3_BUILD
#define GT_API __declspec(dllexport)
#else
#define GT_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GtRunner GtRunner;
typedef struct CUstream_st* GtStream;

// Uses CUDA device 0. The bundle contains schedule.bin, weights.bin and kernels.cubin.
GT_API GtRunner* gt_create(const char* utf8_directory, int batch, int graph);
GT_API void gt_destroy(GtRunner* runner);
GT_API const char* gt_error(void);

// Synchronous host I/O. Both buffers contain batch × 3 × 128 × 128 FP32 values in NCHW order.
GT_API int gt_run(GtRunner* runner, const float* input, float* output);

// The buffers belong to the runner. Populate input before enqueue and synchronize the
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
