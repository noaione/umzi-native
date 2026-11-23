#include "winograd.h"
#include "../../../common/half.h"

#include <array>
#include <cmath>

namespace umzi::vk {
namespace {
constexpr int channels = 32;

float unpack_half(const char* source) {
    uint16_t bits;
    std::memcpy(&bits, source, sizeof(bits));
    const uint32_t sign = uint32_t(bits & 0x8000u) << 16;
    const uint32_t magnitude = bits & 0x7fffu;
    const int exponent = int(magnitude >> 10);
    const uint32_t fraction = magnitude & 0x3ffu;
    if (exponent == 31) {
        const uint32_t encoded = sign | 0x7f800000u | (fraction << 13);
        float special;
        std::memcpy(&special, &encoded, sizeof(special));
        return special;
    }
    const int significand = int(fraction) + (exponent ? 1024 : 0);
    const float value = std::ldexp(float(significand), exponent ? exponent - 25 : -24);
    return sign ? -value : value;
}

// G in G g G^T for a 2x2 output tile and a 3x3 filter.
std::array<float, 4> transform_filter(float a, float b, float c) {
    return {a, (a + b + c) * 0.5f, (a - b + c) * 0.5f, c};
}
}

void transform_winograd_filters(const char* source, char* destination, bool row_major_output) {
    for (int out = 0; out < channels; ++out) {
        for (int in = 0; in < channels; ++in) {
            std::array<float, 4> rows[3];
            for (int y = 0; y < 3; ++y) {
                const char* row = source + sizeof(uint16_t) * ((out * 9 + y * 3) * channels + in);
                const size_t stride = channels * sizeof(uint16_t);
                rows[y] = transform_filter(unpack_half(row), unpack_half(row + stride), unpack_half(row + 2 * stride));
            }
            for (int x = 0; x < 4; ++x) {
                const auto column = transform_filter(rows[0][x], rows[1][x], rows[2][x]);
                for (int y = 0; y < 4; ++y) {
                    const uint16_t half = float_to_half(column[y]);
                    const size_t index = row_major_output
                        ? size_t(((y * channels + in) * channels + out) * 4 + x)
                        : size_t((out * channels + in) * 16 + y * 4 + x);
                    std::memcpy(destination + sizeof(uint16_t) * index, &half, sizeof(half));
                }
            }
        }
    }
}
}
