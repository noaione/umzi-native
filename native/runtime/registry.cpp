#include "kernel.h"

#ifdef UMZI_TEST_KERNEL
#include "../tests/test_kernel.h"
#endif

namespace umzi {
KernelFactory kernel_factory(uint32_t family) {
    switch (family) {
        case gaterv3_id: return make_gaterv3;
        case gfisrv2_id: return make_gfisrv2;
        case gaterv2_id: return make_gaterv2;
        case moesr_id: return make_moesr;
        case smosr_id: return make_smosr;
        case figsr_id: return make_figsr;
        case mosr_id: return make_mosr;
#ifdef UMZI_TEST_KERNEL
        case 0xffff0001u: return make_test_kernel;
#endif
        default: throw std::runtime_error("Umzi kernel family is unavailable: " + std::to_string(family));
    }
}
}
