#include "kernel.h"

namespace umzi::vk {
KernelFactory kernel_factory(uint32_t family) {
    switch (family) {
        case gaterv3_id: return make_gaterv3;
        default: throw std::runtime_error("Vulkan kernel family is unavailable: " + std::to_string(family));
    }
}
}
