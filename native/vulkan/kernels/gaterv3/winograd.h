#pragma once

namespace umzi::vk {
// GateRV3's 32-by-32 3x3 filters are stored output/tap/input. The
// transformed table stores sixteen FP16 values per channel pair, either as
// output/input/coefficient (default) or row/input/output/column for coalesced
// 32-channel loads.
void transform_winograd_filters(const char* source, char* destination, bool row_major_output = false);
}
