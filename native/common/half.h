#pragma once

#include <cstdint>
#include <cstring>

namespace umzi {
// IEEE binary16 with round-to-nearest-even, matching __float2half_rn.
inline uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) return uint16_t(sign | 0x7c00u | (magnitude > 0x7f800000u ? 0x200u : 0u));
    if (magnitude >= 0x477ff000u) return uint16_t(sign | 0x7c00u);
    if (magnitude < 0x38800000u) {
        if (magnitude < 0x33000000u) return uint16_t(sign);
        const uint32_t exponent = magnitude >> 23;
        const uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126 - exponent;
        uint32_t result = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (rest > halfway || (rest == halfway && (result & 1u))) ++result;
        return uint16_t(sign | result);
    }
    uint32_t result = (magnitude - 0x38000000u) >> 13;
    const uint32_t rest = magnitude & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (result & 1u))) ++result;
    return uint16_t(sign | result);
}
}
