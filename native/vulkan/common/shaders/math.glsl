float roundBinary16(float value) {
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

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float mish(float x) {
    float e = exp(-abs(x));
    float square = e * e;
    float numerator = 2.0 * e + (x >= 0.0 ? 1.0 : square);
    return x * numerator / (numerator + (x >= 0.0 ? 2.0 * square : 2.0));
}
