#pragma once

#include "../../common/cuda_plan_cache.h"
#include "../../runtime/model_package.h"

namespace gt {
using umzi::ConvChoice;
using umzi::Digest;
using umzi::GemmChoice;
using umzi::Sha256;
using ShapeKey = umzi::PlanShape;
inline constexpr uint32_t plan_schema = 1;
using PlanCache = umzi::CudaPlanCache<umzi::gaterv3_id, plan_schema>;
}
