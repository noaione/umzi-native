"""Fixed-shape TensorRT-RTX sessions used by both benchmarks."""

from pathlib import Path

import torch


def create_rtx_session(
    path: Path,
    batch: int,
    tile: int,
    cache_dir: Path,
    profile_prefix: Path | None = None,
):
    import onnxruntime as ort
    import onnxruntime_ep_nv_tensorrt_rtx as rtx

    ort.disable_telemetry_events()
    ep = rtx.get_ep_name()
    devices = [device for device in ort.get_ep_devices() if device.ep_name == ep]
    if not devices:
        ort.register_execution_provider_library(ep, rtx.get_library_path())
        devices = [device for device in ort.get_ep_devices() if device.ep_name == ep]
    if not devices:
        raise RuntimeError(f"No execution-provider device is available for {ep}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    profile = f"input:{batch}x3x{tile}x{tile}"
    config = {
        "enable_cuda_graph": "1",
        "nv_max_workspace_size": str(torch.cuda.get_device_properties(0).total_memory * 3 // 4),
        "nv_detailed_build_log": "0",
        "nv_runtime_cache_path": str(cache_dir.resolve()),
        "nv_profile_min_shapes": profile,
        "nv_profile_opt_shapes": profile,
        "nv_profile_max_shapes": profile,
    }
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices([devices[0]], config)
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    session = ort.InferenceSession(str(path), sess_options=options, enable_fallback=0)
    if ep not in session.get_providers():
        raise RuntimeError(f"TensorRT-RTX did not activate: {session.get_providers()}")
    if list(session.get_inputs()[0].shape) != [batch, 3, tile, tile]:
        raise RuntimeError(f"Unexpected TensorRT-RTX input shape for {path}")
    return session, config
