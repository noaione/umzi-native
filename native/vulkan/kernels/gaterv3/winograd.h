#pragma once

namespace umzi::vk {
// GateRV3's 32-by-32 3x3 filters are stored output/tap/input. The
// transformed table is output/input/coefficient with sixteen FP16 values.
void transform_winograd_filters(const char* source, char* destination);
}
