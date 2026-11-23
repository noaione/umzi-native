#include "buffers.glsl"
#include "math.glsl"

layout(push_constant, std430) uniform Schedule {
    ivec4 offsets;
    ivec4 channels;
    ivec4 weights;
    int outputOffset;
    int outputChannels;
    int h;
    int w;
    int batch;
    int kind;
    int inputH;
    int inputW;
    int kernelSize;
    int channelOffset;
    int mixed;
    int scratch;
} op;

float rounded(float value) {
    return op.mixed == 0 ? value : roundBinary16(value);
}

void storeActivation(int index, float value) {
#ifdef HALF_STORAGE
    data[index] = float16_t(value);
#else
    data[index] = rounded(value);
#endif
}
