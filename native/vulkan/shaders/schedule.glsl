#ifdef HALF_STORAGE
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define Activation float16_t
layout(set = 0, binding = 2, std430) buffer PlanarRgb { float rgb[]; };
#else
#define Activation float
#define rgb data
#endif
layout(set = 0, binding = 0, std430) buffer Activations { Activation data[]; };
layout(set = 0, binding = 1, std430) readonly buffer Weights { uint packed[]; };
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

int groupIndex() {
    return int(gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.y);
}

float weight(int offset, int index) {
    if (offset < 0) return 0.0;
    vec2 values = unpackHalf2x16(packed[(offset >> 2) + (index >> 1)]);
    return values[index & 1];
}

float rounded(float value) {
    if (op.mixed == 0) return value;
    // Keep the binary16 rounding boundary explicit even on drivers that
    // optimize away an immediately unpacked half conversion.
    uint bits = floatBitsToUint(value);
    uint signBit = bits & 0x80000000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) return value;
    if (magnitude < 0x38800000u) { // Binary16 subnormals are multiples of 2^-24.
        float subnormal = roundEven(uintBitsToFloat(magnitude) * 16777216.0) * (1.0 / 16777216.0);
        return uintBitsToFloat(signBit | floatBitsToUint(subnormal));
    }
    // Round the 13 discarded mantissa bits to nearest, with ties to even.
    magnitude = (magnitude + 0xfffu + ((magnitude >> 13) & 1u)) & 0xffffe000u;
    return uintBitsToFloat(signBit | (magnitude >= 0x47800000u ? 0x7f800000u : magnitude));
}

float loadActivation(int index) {
    return float(data[index]);
}

void storeActivation(int index, float value) {
#ifdef HALF_STORAGE
    data[index] = float16_t(value);
#else
    data[index] = rounded(value);
#endif
}

vec2 weightPair(int offset, int index) {
    return unpackHalf2x16(packed[(offset >> 2) + (index >> 1)]);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float mish(float x) {
    float e = exp(-abs(x));
    float square = e * e;
    float numerator = 2.0 * e + (x >= 0.0 ? 1.0 : square);
    return x * numerator / (numerator + (x >= 0.0 ? 2.0 * square : 2.0));
}
