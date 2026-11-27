#pragma once

#include <stdint.h>

#ifdef _WIN32
#  ifdef UMZI_BUILD
#    define UMZI_API __declspec(dllexport)
#  else
#    define UMZI_API __declspec(dllimport)
#  endif
#else
#  define UMZI_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct UmziSession UmziSession;
typedef struct CUstream_st* UmziStream;
typedef enum UmziStatus {
    UMZI_OK = 0, UMZI_ERROR = -1, UMZI_OUT_OF_MEMORY = -2
} UmziStatus;
typedef enum UmziPrecision {
    UMZI_MIXED = 0, UMZI_FP16 = 1, UMZI_FP32 = 2
} UmziPrecision;
typedef enum UmziDtype {
    UMZI_FLOAT32 = 1, UMZI_FLOAT16 = 2
} UmziDtype;
typedef enum UmziSessionFlags {
    UMZI_CUDA_GRAPH = 1
} UmziSessionFlags;
typedef enum UmziCacheStatus {
    UMZI_CACHE_DISABLED, UMZI_CACHE_MISS, UMZI_CACHE_HIT, UMZI_CACHE_REBUILT, UMZI_CACHE_UNAVAILABLE
} UmziCacheStatus;

typedef struct UmziOption {
    const char* name;
    int64_t value;
} UmziOption;

typedef struct UmziSessionOptions {
    uint32_t struct_size;
    int32_t device;
    uint32_t precision, flags;
    const char* cache_dir;
    const UmziOption* kernel_options;
    uint32_t kernel_option_count, reserved;
} UmziSessionOptions;

typedef struct UmziShape {
    uint32_t rank;
    uint32_t dimensions[4];
} UmziShape;

typedef struct UmziTensor {
    const char* name;
    UmziShape shape;  // Zero dimensions in metadata are dynamic.
    uint32_t dtype;
    uint64_t bytes;   // Set by describe for a concrete shape.
} UmziTensor;

typedef struct UmziIo {
    uint32_t struct_size;
    UmziTensor input, output;
} UmziIo;

typedef struct UmziModelInfo {
    uint32_t struct_size;
    const char* kernel;
    uint32_t scale, context;
} UmziModelInfo;

typedef struct UmziSessionInfo {
    uint32_t struct_size, cache_status;
    UmziShape prepared_input; // Padded shape, rank 0 before preparation.
    uint64_t buffer_bytes[4];  // Weights, activations, workspace, I/O.
    uint64_t startup_device_used_bytes[4];
    double model_load_ms;
    double shape_setup_ms[4];  // Allocation, plans, warmup, graph capture.
} UmziSessionInfo;

typedef struct UmziDiagnostic {
    uint32_t struct_size;
    const char* name;
    int64_t value;
} UmziDiagnostic;

typedef enum UmziPageFormat { UMZI_PAGE_U8 = 0, UMZI_PAGE_FLOAT32 = 1 } UmziPageFormat;
typedef enum UmziOverlapMethod { UMZI_OVERLAP_CENTERED = 0, UMZI_OVERLAP_LINEAR = 1 } UmziOverlapMethod;
typedef enum UmziTileLayout { UMZI_TILE_DEFAULT = 0, UMZI_TILE_EVEN = 1, UMZI_TILE_FEWEST_EVEN = 2 } UmziTileLayout;

typedef struct UmziTilingOptions {
    uint32_t struct_size, tile_size, batch_size, overlap;
    uint32_t background, output_format;
    uint32_t overlap_method;
    uint32_t layout;  // UMZI_TILE_DEFAULT selects the entry point's original layout.
} UmziTilingOptions;

UMZI_API uint32_t umzi_abi_version(void);
UMZI_API const char* umzi_error(void);
UMZI_API int umzi_error_code(void);
// Caller initializes struct_size. Strings returned by the API belong to the session unless noted.
// umzi_session_cache_message and umzi_session_kernel_message return a pointer that stays valid only until the
// next call on the same session (including another message call); copy the text if it must outlive that.
// umzi_error is per thread and valid until the next API call on that thread.
UMZI_API UmziSession* umzi_session_open(const char* utf8_model, const UmziSessionOptions* options);
UMZI_API void umzi_session_close(UmziSession* session);
UMZI_API int umzi_session_model(const UmziSession* session, UmziModelInfo* info);
UMZI_API int umzi_session_tensors(const UmziSession* session, UmziIo* io);
UMZI_API int umzi_session_describe(const UmziSession* session, const UmziShape* shape, UmziIo* io);
UMZI_API int umzi_session_prepare(UmziSession* session, const UmziShape* shape);
UMZI_API int umzi_session_run(UmziSession* session, const UmziShape* shape,
                              const void* input, uint64_t input_bytes, void* output, uint64_t output_bytes);
// Serialize calls per session. CUDA buffers and the stream must belong to the selected device's primary context.
// Synchronize the stream before reusing the caller's buffers. Close waits for pending work.
UMZI_API int umzi_session_enqueue(UmziSession* session, const UmziShape* shape,
                                  uint64_t input, uint64_t input_bytes, uint64_t output,
                                  uint64_t output_bytes, UmziStream stream);
// Synchronous page inference. The buffers are contiguous HWC at input and scaled output size.
UMZI_API int umzi_session_run_tiled(UmziSession* session, uint32_t height, uint32_t width,
                                    const UmziTilingOptions* options, const uint8_t* input,
                                    uint64_t input_bytes, void* output, uint64_t output_bytes);
UMZI_API int umzi_session_run_tiled_variable(UmziSession* session, uint32_t height, uint32_t width,
                                             const UmziTilingOptions* options, const uint8_t* input,
                                             uint64_t input_bytes, void* output, uint64_t output_bytes);
UMZI_API int umzi_session_info(const UmziSession* session, UmziSessionInfo* info);
UMZI_API uint32_t umzi_session_diagnostic_count(const UmziSession* session);
UMZI_API int umzi_session_diagnostic(const UmziSession* session, uint32_t index, UmziDiagnostic* diagnostic);
UMZI_API const char* umzi_session_cache_message(const UmziSession* session);
UMZI_API const char* umzi_session_kernel_message(const UmziSession* session);
UMZI_API int umzi_session_profile(UmziSession* session, float* milliseconds, uint32_t count);

#ifdef __cplusplus
}
#endif

#undef UMZI_API
