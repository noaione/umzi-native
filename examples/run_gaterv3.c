#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include "umzi_native.h"

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) {
        fputs("Usage: run_gaterv3 input.ummodel\n", stderr);
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

    UmziSession* session = umzi_session_open(path, NULL);
    free(path);
    if (!session) {
        fprintf(stderr, "%s\n", umzi_error());
        return 1;
    }

    const uint32_t batch = 3, height = 63, width = 79;
    UmziShape shape = {4, {batch, 3, height, width}};
    UmziIo io = {0};
    io.struct_size = sizeof(io);
    int status = umzi_session_describe(session, &shape, &io);
    if (!status) {
        void* input = calloc(1, (size_t)io.input.bytes);
        void* output = malloc((size_t)io.output.bytes);
        if (input && output) {
            status = umzi_session_run(session, &shape, input, io.input.bytes, output, io.output.bytes);
            if (!status)
                printf("Restored %u tiles of %ux%u, first RGB sample %.6f\n", batch, height, width, ((float*)output)[0]);
        } else {
            fputs("Could not allocate caller buffers\n", stderr);
            status = UMZI_OUT_OF_MEMORY;
        }
        free(output);
        free(input);
    }
    if (status == UMZI_ERROR || status == UMZI_OUT_OF_MEMORY)
        fprintf(stderr, "%s\n", umzi_error());
    umzi_session_close(session);
    return status ? 1 : 0;
}
