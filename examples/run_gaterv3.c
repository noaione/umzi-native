#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include "runner.h"

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) {
        fputs("Usage: run_gaterv3 input.gtmodel\n", stderr);
        return 2;
    }

    int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[1], -1, NULL, 0, NULL, NULL);
    if (!length) return 2;
    char* path = malloc(length);
    if (!path) return 2;
    if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[1], -1, path, length, NULL, NULL)) {
        free(path);
        return 2;
    }

    GtSessionOptions options = {sizeof(options), 0, GT_CREATE_GRAPH, 0, NULL};
    GtSession* session = gt_session_open(path, &options);
    free(path);
    if (!session) {
        fprintf(stderr, "%s\n", gt_error());
        return 1;
    }

    const uint32_t batch = 3, height = 63, width = 79;
    GtIo io = {0};
    io.struct_size = sizeof(io);
    int status = gt_session_describe(session, batch, height, width, &io);
    if (!status) {
        void* input = calloc(1, (size_t)io.input_bytes);
        void* output = malloc((size_t)io.output_bytes);
        if (input && output) {
            status = gt_session_run(session, batch, height, width, input, io.input_bytes, output, io.output_bytes);
            if (!status)
                printf("Restored %u tiles of %ux%u, first RGB sample %.6f\n", batch, height, width, ((float*)output)[0]);
        } else {
            fputs("Could not allocate caller buffers\n", stderr);
            status = GT_STATUS_OUT_OF_MEMORY;
        }
        free(output);
        free(input);
    }
    if (status == GT_STATUS_ERROR || status == GT_STATUS_OUT_OF_MEMORY)
        fprintf(stderr, "%s\n", gt_error());
    gt_session_close(session);
    return status ? 1 : 0;
}
