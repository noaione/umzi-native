#if defined(HALF_STORAGE) || defined(HALF_MATH)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#ifdef HALF_STORAGE
#extension GL_EXT_shader_16bit_storage : require
#define Activation float16_t
layout(set = 0, binding = 2, std430) buffer PlanarRgb { float rgb[]; };
#else
#define Activation float
#define rgb data
#endif
layout(set = 0, binding = 0, std430) buffer Activations { Activation data[]; };
layout(set = 0, binding = 1, std430) readonly buffer Weights { uint packed[]; };

int groupIndex() {
    return int(gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.y);
}

float weight(int offset, int index) {
    if (offset < 0) return 0.0;
    vec2 values = unpackHalf2x16(packed[(offset >> 2) + (index >> 1)]);
    return values[index & 1];
}

vec2 weightPair(int offset, int index) {
    return unpackHalf2x16(packed[(offset >> 2) + (index >> 1)]);
}

float loadActivation(int index) {
    return float(data[index]);
}
