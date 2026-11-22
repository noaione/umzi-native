"""Matched 128x128 full-page benchmark for GFISRv2 native sessions.

Follows the umzi-bench unified-matrix contract on samples/mscale/lb-page-003.png:
- batch 1, 128x128 input tiles, 32 px input overlap, 4x output (150 tiles, 3600x5400)
- row-major 96 px grid, replicate edge padding, crop, normalized linear feather stitch
- 2 CPU threads, TF32 disabled, CUDA Graph replay, one GPU worker per subprocess
- 4 alternating forward/reverse timing rounds after excluded full-page warmup
- separate isolated NVML VRAM workers (steady peak above idle + native session buffers)
- separate per-tile synchronized batch-1 latency measurement (all 150 tiles saved)
- CPU scoring against fresh untouched eager CUDA FP32 (raw PSNR + RGB16/grayscale SSIMULACRA2)
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark_architectures import METRIC, ROOT, digest, ssimulacra2
from PIL import Image
from reference.native_session import DeviceSession
from torch import Tensor
from torch.nn import functional

from umzi_native import InferenceSession, SessionOptions
from umzi_native.exporter import export_model
from umzi_native.exporter._loader import load_checkpoint

CHECKPOINT = ROOT / "models/4x_wtp_MangaScale_GfisrV2.pth"
PAGE_PATH = ROOT / "samples/mscale/lb-page-003.png"
EXPECTED_PAGE_SHA256 = "528addf2f7cbe2de28fe1ae8f17135779c96ac7b62e5ed4322c697370c6f50da"
TILE = 128
OVERLAP = 32
SCALE = 4
MIB = 2**20


@dataclass(frozen=True)
class RunnerSpec:
    name: str
    kind: str  # "eager-fp32", "eager-fp16", or "native"
    precision: str
    dll_path: Path | None
    kernel_options: dict[str, int]
    cache_dir: Path | None


def parse_runner_spec(raw: str) -> RunnerSpec:
    if raw in ("eager-fp32", "eager-fp16"):
        return RunnerSpec(raw, raw, "fp32" if raw == "eager-fp32" else "fp16", None, {}, None)
    if "=" not in raw:
        raise ValueError(f"Runner spec must be 'eager-fp32', 'eager-fp16', or 'name=precision:dll[:k=v,...]': {raw}")
    name, body = raw.split("=", 1)
    parts = body.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"Native runner spec requires precision and dll path: {raw}")
    precision = parts[0]
    if precision not in ("fp32", "fp16", "mixed"):
        raise ValueError(f"Unsupported precision {precision!r} in {raw}")
    # Handle Windows drive letters (e.g. C:\path or I:\path) in dll_path.
    rest = body[len(precision) + 1 :]
    options: dict[str, int] = {}
    cache_dir: Path | None = None
    dll_str = rest
    if "|" in rest:
        dll_str, opt_str = rest.split("|", 1)
        for token in opt_str.split(","):
            token = token.strip()
            if not token:
                continue
            key, value = token.split("=", 1)
            if key == "cache_dir":
                cache_dir = Path(value).resolve()
            else:
                options[key] = int(value)
    dll_path = Path(dll_str).resolve()
    if not dll_path.is_file():
        raise FileNotFoundError(f"Native DLL not found: {dll_path}")
    return RunnerSpec(name, "native", precision, dll_path, options, cache_dir)


def serialize_spec(spec: RunnerSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "kind": spec.kind,
        "precision": spec.precision,
        "dll_path": str(spec.dll_path) if spec.dll_path else None,
        "dll_sha256": digest(spec.dll_path) if spec.dll_path else None,
        "kernel_options": spec.kernel_options,
        "cache_dir": str(spec.cache_dir) if spec.cache_dir else None,
    }


def load_spec(data: dict[str, Any]) -> RunnerSpec:
    return RunnerSpec(
        name=data["name"],
        kind=data["kind"],
        precision=data["precision"],
        dll_path=Path(data["dll_path"]) if data["dll_path"] else None,
        kernel_options=dict(data["kernel_options"]),
        cache_dir=Path(data["cache_dir"]) if data["cache_dir"] else None,
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def image_to_tensor(image: Image.Image) -> Tensor:
    data = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_to_image(tensor: Tensor, template: Image.Image) -> Image.Image:
    data = tensor[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    image = Image.fromarray(data)
    if template.mode in {"1", "L", "I", "F", "I;16"}:
        image = image.convert("L")
    return image


def write_png16_rgb(tensor: Tensor, path: Path) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Cannot render nonfinite output")
    array = tensor[0].clamp(0, 1).mul(65535).round().to(torch.int32).permute(1, 2, 0).numpy()
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError("Require RGB")
    rows = np.ascontiguousarray(array, dtype=">u2").view(np.uint8).reshape(height, width * 6)
    filtered = np.zeros((height, width * 6 + 1), dtype=np.uint8)
    filtered[:, 1:] = rows

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0))
        + chunk(b"sRGB", b"\x00")
        + chunk(b"IDAT", zlib.compress(filtered.tobytes(), 1))
        + chunk(b"IEND", b"")
    )


def infer_tensor(
    tensor: Tensor,
    runner: Callable[[Tensor], Tensor],
    *,
    scale: int = SCALE,
    tile: int = TILE,
    overlap: int = OVERLAP,
) -> Tensor:
    """Match umzi-bench/src/janai/inference.py infer_tensor contract exactly."""
    tensor = tensor.cpu()
    h, w = tensor.shape[-2:]
    out = torch.zeros(1, 3, h * scale, w * scale)
    weights = torch.zeros(1, 1, h * scale, w * scale) if overlap else None
    step = tile - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = tensor[:, :, y : min(y + tile, h), x : min(x + tile, w)]
            ph, pw = patch.shape[-2:]
            padded = functional.pad(patch, (0, tile - pw, 0, tile - ph), mode="replicate")
            prediction = runner(padded).cpu()[:, :, : ph * scale, : pw * scale]
            if prediction.shape != (1, 3, ph * scale, pw * scale):
                raise ValueError(f"Runner returned unexpected output shape {prediction.shape}")
            sy, sx, eh, ew = y * scale, x * scale, ph * scale, pw * scale
            target = (slice(None), slice(None), slice(sy, sy + eh), slice(sx, sx + ew))
            if weights is None:
                out[target] = prediction
            else:
                wy, wx = torch.ones(eh), torch.ones(ew)
                n = min(overlap * scale, eh)
                if y:
                    wy[:n] *= torch.arange(1, n + 1) / (n + 1)
                if y + ph < h:
                    wy[-n:] *= torch.arange(n, 0, -1) / (n + 1)
                n = min(overlap * scale, ew)
                if x:
                    wx[:n] *= torch.arange(1, n + 1) / (n + 1)
                if x + pw < w:
                    wx[-n:] *= torch.arange(n, 0, -1) / (n + 1)
                blend = wy[:, None] * wx[None, :]
                out[target] += prediction * blend
                weights[:, :, sy : sy + eh, sx : sx + ew] += blend
    if weights is not None:
        out.div_(weights)
    return out


class NativeHostRunner:
    """Adapt umzi_native.InferenceSession to the CPU-tensor infer_tensor contract."""

    def __init__(self, session: InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.np_dtype = np.float16 if session.get_inputs()[0].type == "tensor(float16)" else np.float32
        self.torch_dtype = torch.float16 if self.np_dtype == np.float16 else torch.float32

    def __call__(self, tensor: Tensor) -> Tensor:
        source = tensor.to(dtype=self.torch_dtype).contiguous().numpy()
        (result,) = self.session.run(None, {self.input_name: source})
        return torch.from_numpy(result).float()


class TorchHostRunner:
    def __init__(self, model: torch.nn.Module, dtype: torch.dtype):
        self.model = model.eval().to(device="cuda", dtype=dtype)
        self.dtype = dtype

    @torch.inference_mode()
    def __call__(self, tensor: Tensor) -> Tensor:
        return self.model(tensor.to(device="cuda", dtype=self.dtype)).float().cpu()


class _MemoryInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


class Nvml:
    def __init__(self) -> None:
        self.library = ctypes.CDLL("C:/Windows/System32/nvml.dll")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.library.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemoryInfo)]
        self._check(self.library.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self._check(self.library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))

    @staticmethod
    def _check(status: int) -> None:
        if status:
            raise RuntimeError(f"NVML memory query failed: {status}")

    def read(self) -> dict[str, float]:
        info = _MemoryInfo()
        self._check(self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)))
        return {"used_mib": info.used / MIB, "total_mib": info.total / MIB}

    def close(self) -> None:
        self._check(self.library.nvmlShutdown())


class NvmlSampler:
    def __init__(self, nvml: Nvml, interval: float = 0.005) -> None:
        self.nvml = nvml
        self.interval = interval
        self.samples: list[list[float]] = []
        self.done = threading.Event()
        self.error: Exception | None = None
        self.start = 0.0
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        self.samples.append([time.perf_counter() - self.start, self.nvml.read()["used_mib"]])

    def _loop(self) -> None:
        try:
            while not self.done.wait(self.interval):
                self._sample()
        except Exception as error:
            self.error = error

    def __enter__(self) -> NvmlSampler:
        self.start = time.perf_counter()
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.done.set()
        if self.thread is not None:
            self.thread.join()
        self._sample()
        if self.error is not None:
            raise self.error

    def report(self, background: float) -> dict[str, Any]:
        values = [value for _, value in self.samples]
        return {
            "device_peak_mib": max(values),
            "device_median_mib": statistics.median(values),
            "peak_above_pre_context_background_mib": max(values) - background,
            "samples": len(values),
            "requested_sampling_interval_ms": self.interval * 1000,
        }


def configure_runtime() -> None:
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def git_revision() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    return subprocess.check_output([git, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def open_native_session(
    spec: RunnerSpec, package: Path, *, captured: bool, default_cache_dir: Path | None = None
) -> InferenceSession:
    cache_dir = spec.cache_dir if spec.cache_dir is not None else default_cache_dir
    options = SessionOptions(
        precision=spec.precision,
        dll_path=spec.dll_path,
        enable_cuda_graph=captured,
        cache_dir=cache_dir,
        kernel_options=spec.kernel_options,
    )
    session = InferenceSession(package, options)
    session.prepare((1, 3, TILE, TILE))
    return session


def measure_tile_latencies(
    page: Tensor,
    runner_device: Callable[[Tensor], Tensor],
    dtype: torch.dtype,
    image_name: str,
) -> list[dict[str, Any]]:
    h, w = page.shape[-2:]
    step = TILE - OVERLAP
    samples: list[dict[str, Any]] = []
    index = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = page[:, :, y : min(y + TILE, h), x : min(x + TILE, w)]
            ph, pw = patch.shape[-2:]
            padded = functional.pad(patch, (0, TILE - pw, 0, TILE - ph), mode="replicate")
            gpu_tile = padded.to(device="cuda", dtype=dtype).contiguous()
            torch.cuda.synchronize()
            started = time.perf_counter()
            _ = runner_device(gpu_tile)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            samples.append({
                "image": image_name,
                "tile_index": index,
                "top": y,
                "left": x,
                "milliseconds": elapsed_ms,
            })
            index += 1
    return samples


def measure_host_tile_latencies(
    page: Tensor,
    runner_host: Callable[[Tensor], Tensor],
    image_name: str,
) -> list[dict[str, Any]]:
    h, w = page.shape[-2:]
    step = TILE - OVERLAP
    samples: list[dict[str, Any]] = []
    index = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = page[:, :, y : min(y + TILE, h), x : min(x + TILE, w)]
            ph, pw = patch.shape[-2:]
            padded = functional.pad(patch, (0, TILE - pw, 0, TILE - ph), mode="replicate")
            started = time.perf_counter()
            _ = runner_host(padded)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            samples.append({
                "image": image_name,
                "tile_index": index,
                "top": y,
                "left": x,
                "milliseconds": elapsed_ms,
            })
            index += 1
    return samples


def runtime_metadata() -> dict[str, Any]:
    return {
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cpu_threads": torch.get_num_threads(),
        "tf32": False,
    }


@torch.inference_mode()
def run_worker(args: argparse.Namespace) -> None:
    configure_runtime()
    directory = args.output.resolve()
    definition = json.loads((directory / "definition.json").read_text(encoding="utf-8"))
    spec = load_spec(definition["runners"][args.worker])
    package = Path(definition["package"]["path"])
    with Image.open(PAGE_PATH) as img:
        page = image_to_tensor(img)
    h, w = page.shape[-2:]
    input_pixels = h * w

    default_cache = directory / "plans" / spec.name if spec.kind == "native" and spec.precision != "fp32" else None
    if args.mode == "vram":
        nvml = Nvml()
        try:
            before = nvml.read()
            session_info: dict[str, Any] | None = None
            warmed_delta_bytes = 0
            with NvmlSampler(nvml) as startup:
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats()
                free_before = torch.cuda.mem_get_info()[0]
                if spec.kind == "native":
                    session = open_native_session(spec, package, captured=True, default_cache_dir=default_cache)
                    runner: Callable[[Tensor], Tensor] = NativeHostRunner(session)
                    session_info = session.get_session_info()
                else:
                    session = None
                    model, _ = load_checkpoint(CHECKPOINT)
                    dtype = torch.float32 if spec.kind == "eager-fp32" else torch.float16
                    runner = TorchHostRunner(model, dtype)
                torch.cuda.synchronize()
                warmup = infer_tensor(page, runner)
                torch.cuda.synchronize()
                if not bool(torch.isfinite(warmup).all()):
                    raise ValueError("VRAM warmup produced nonfinite output")
                del warmup
                torch.cuda.empty_cache()
                warmed_delta_bytes = free_before - torch.cuda.mem_get_info()[0]
            startup_torch = {
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
            }
            gc.collect()
            torch.cuda.synchronize()
            resident = nvml.read()
            torch.cuda.reset_peak_memory_stats()
            with NvmlSampler(nvml) as steady:
                actual = infer_tensor(page, runner)
                torch.cuda.synchronize()
            if not bool(torch.isfinite(actual).all()):
                raise ValueError("VRAM inference produced nonfinite output")
            del actual
            free_after, total_mem = torch.cuda.mem_get_info()
            result: dict[str, Any] = {
                "runner": spec.name,
                "pre_context_device_background_mib": before["used_mib"],
                "device_total_mib": before["total_mib"],
                "after_warmup_device_used_mib": resident["used_mib"],
                "warmed_device_vram_delta_mib": warmed_delta_bytes / MIB,
                "after_inference_cuda_mem_get_info_used_mib": (total_mem - free_after) / MIB,
                "startup_compile_warmup": {**startup.report(before["used_mib"]), "torch": startup_torch},
                "steady_full_page": {
                    **steady.report(before["used_mib"]),
                    "torch": {
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
                    },
                },
                "session_info": session_info,
                "runtime": runtime_metadata(),
            }
            if session is not None:
                session.close()
        finally:
            nvml.close()
        write_json(args.result, result)
        return

    if args.mode == "quality":
        started = time.perf_counter()
        if spec.kind == "native":
            with open_native_session(spec, package, captured=True, default_cache_dir=default_cache) as graph_session:
                setup_s = time.perf_counter() - started
                host_runner = NativeHostRunner(graph_session)
                actual = infer_tensor(page, host_runner)
                if not bool(torch.isfinite(actual).all()):
                    raise ValueError(f"Non-finite full-page output for {spec.name}")
                session_info = graph_session.get_session_info()
                # Measure individual synchronized batch-1 latencies (device-resident and host-session).
                device_runner = DeviceSession(graph_session)
                _ = device_runner(torch.zeros((1, 3, TILE, TILE), dtype=device_runner.dtype, device="cuda"))
                torch.cuda.synchronize()
                device_latencies = measure_tile_latencies(page, device_runner, device_runner.dtype, PAGE_PATH.name)
                host_latencies = measure_host_tile_latencies(page, host_runner, PAGE_PATH.name)
            with open_native_session(spec, package, captured=False, default_cache_dir=default_cache) as direct_session:
                direct_actual = infer_tensor(page, NativeHostRunner(direct_session))
            graph_direct_bit_identical = np.array_equal(
                actual.numpy().view(np.int32), direct_actual.numpy().view(np.int32)
            )
            if not graph_direct_bit_identical:
                raise AssertionError(f"CUDA Graph and direct execution outputs differ for {spec.name}")
        else:
            model, _ = load_checkpoint(CHECKPOINT)
            dtype = torch.float32 if spec.kind == "eager-fp32" else torch.float16
            torch_runner = TorchHostRunner(model, dtype)
            setup_s = time.perf_counter() - started
            actual = infer_tensor(page, torch_runner)
            if not bool(torch.isfinite(actual).all()):
                raise ValueError(f"Non-finite full-page output for {spec.name}")
            session_info = None
            graph_direct_bit_identical = None
            _ = torch_runner.model(torch.zeros((1, 3, TILE, TILE), dtype=dtype, device="cuda"))
            torch.cuda.synchronize()
            device_latencies = measure_tile_latencies(page, torch_runner.model, dtype, PAGE_PATH.name)
            host_latencies = measure_host_tile_latencies(page, torch_runner, PAGE_PATH.name)

        raw = actual.numpy()
        raw_path = directory / "raw" / f"{spec.name}.npy"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, raw)
        result = {
            "runner": spec.name,
            "setup_seconds": setup_s,
            "output_shape": list(raw.shape),
            "output_sha256": array_sha256(raw),
            "graph_direct_bit_identical": graph_direct_bit_identical,
            "real_tiles": len(device_latencies),
            "average_batch1_device_tile_latency_ms": statistics.mean(x["milliseconds"] for x in device_latencies),
            "median_batch1_device_tile_latency_ms": statistics.median(x["milliseconds"] for x in device_latencies),
            "average_batch1_host_tile_latency_ms": statistics.mean(x["milliseconds"] for x in host_latencies),
            "median_batch1_host_tile_latency_ms": statistics.median(x["milliseconds"] for x in host_latencies),
            "batch1_device_tile_latency": device_latencies,
            "batch1_host_tile_latency": host_latencies,
            "session_info": session_info,
            "runtime": runtime_metadata(),
        }
        write_json(args.result, result)
        return

    if args.mode == "timing":
        started = time.perf_counter()
        if spec.kind == "native":
            session = open_native_session(spec, package, captured=True, default_cache_dir=default_cache)
            runner = NativeHostRunner(session)
            session_info = session.get_session_info()
        else:
            session = None
            model, _ = load_checkpoint(CHECKPOINT)
            dtype = torch.float32 if spec.kind == "eager-fp32" else torch.float16
            runner = TorchHostRunner(model, dtype)
            session_info = None
        setup_s = time.perf_counter() - started
        try:
            # Excluded full-page warmup.
            _ = infer_tensor(page, runner)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            actual = infer_tensor(page, runner)
            elapsed = time.perf_counter() - t0
        finally:
            if session is not None:
                session.close()
        page_sha = array_sha256(actual.numpy())
        quality_report = json.loads((directory / "quality" / f"{spec.name}.json").read_text(encoding="utf-8"))
        if page_sha != quality_report["output_sha256"]:
            raise AssertionError(f"Timing output hash changed for {spec.name}")
        result = {
            "runner": spec.name,
            "setup_seconds": setup_s,
            "page_seconds": elapsed,
            "input_pixels": input_pixels,
            "page_input_mpx_per_second": input_pixels / elapsed / 1e6,
            "page_sha256": page_sha,
            "session_info": session_info,
            "runtime": runtime_metadata(),
        }
        write_json(args.result, result)
        return

    raise ValueError(f"Unsupported worker mode: {args.mode}")


def score_raw(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError(f"Output shape {candidate.shape} differs from {reference.shape}")
    squared_error = 0.0
    max_abs = 0.0
    nonfinite = 0
    thresholds = {
        "half_8bit_step": 0.5 / 255.0,
        "one_8bit_step": 1.0 / 255.0,
        "two_8bit_steps": 2.0 / 255.0,
        "four_8bit_steps": 4.0 / 255.0,
    }
    counts = dict.fromkeys(thresholds, 0)
    clipped = 0
    for start in range(0, reference.shape[-2], 128):
        a = reference[..., start : start + 128, :]
        b = candidate[..., start : start + 128, :]
        nonfinite += int(b.size - np.count_nonzero(np.isfinite(b)))
        delta = np.subtract(b, a, dtype=np.float64)
        squared_error += float(np.sum(np.square(delta)))
        abs_delta = np.abs(delta)
        max_abs = max(max_abs, float(np.max(abs_delta)))
        worst_rgb = np.max(abs_delta, axis=1)
        for label, thr in thresholds.items():
            counts[label] += int(np.count_nonzero(worst_rgb <= thr))
        clipped += int(np.count_nonzero((b < 0.0) | (b > 1.0)))
    if nonfinite:
        return {"finite": False, "nonfinite_values": nonfinite}
    mse = squared_error / reference.size
    pixels = reference.shape[0] * reference.shape[-2] * reference.shape[-1]
    return {
        "finite": True,
        "values": reference.size,
        "mse": mse,
        "psnr_db": -10.0 * math.log10(mse) if mse else None,
        "max_abs": max_abs,
        "pixel_fraction_within": {k: v / pixels for k, v in counts.items()},
        "candidate_out_of_range_channel_fraction": clipped / reference.size,
    }


def run_scoring(directory: Path, reference_npy: Path, runner_names: list[str]) -> dict[str, Any]:
    with Image.open(PAGE_PATH) as img:
        template = img.copy()
    ref_arr = np.load(reference_npy, mmap_mode="r")
    ref_tensor = torch.from_numpy(np.array(ref_arr, copy=True))
    ref_dir = directory / "renders" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref16 = ref_dir / "reference-rgb16.png"
    ref8 = ref_dir / "reference-display.png"
    if not ref16.is_file():
        write_png16_rgb(ref_tensor, ref16)
    if not ref8.is_file():
        tensor_to_image(ref_tensor, template).save(ref8)
    scores: dict[str, Any] = {}
    for name in runner_names:
        cand_arr = np.load(directory / "raw" / f"{name}.npy", mmap_mode="r")
        raw = score_raw(ref_arr, cand_arr)
        cand_dir = directory / "renders" / name
        cand_dir.mkdir(parents=True, exist_ok=True)
        cand_tensor = torch.from_numpy(np.array(cand_arr, copy=True))
        out16 = cand_dir / "candidate-rgb16.png"
        out8 = cand_dir / "candidate-display.png"
        write_png16_rgb(cand_tensor, out16)
        tensor_to_image(cand_tensor, template).save(out8)
        raw["ssimulacra2_rgb16"] = ssimulacra2(ref16, out16)
        raw["ssimulacra2_display8"] = ssimulacra2(ref8, out8)
        raw["display_mode"] = template.mode
        scores[name] = raw
        psnr_text = "inf" if raw["psnr_db"] is None else f"{raw['psnr_db']:.4f}"
        print(
            f"  score {name}: PSNR {psnr_text} dB, gray SSIMULACRA2 {raw['ssimulacra2_display8']:.4f}, "
            f"RGB16 SSIMULACRA2 {raw['ssimulacra2_rgb16']:.4f}, max_abs {raw['max_abs']:.6g}",
            flush=True,
        )
    write_json(
        directory / "quality-report.json", {"reference": str(reference_npy), "metric": str(METRIC), "runners": scores}
    )
    return scores


def ensure_reference(reference_dir: Path) -> Path:
    reference_dir.mkdir(parents=True, exist_ok=True)
    ref_npy = reference_dir / "eager-fp32.npy"
    meta_path = reference_dir / "meta.json"
    page_sha = digest(PAGE_PATH)
    ckpt_sha = digest(CHECKPOINT)
    if page_sha != EXPECTED_PAGE_SHA256:
        raise ValueError(f"Unexpected page SHA-256 {page_sha} != {EXPECTED_PAGE_SHA256}")
    if ref_npy.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("checkpoint_sha256") == ckpt_sha
            and meta.get("page_sha256") == page_sha
            and meta.get("tile") == TILE
            and meta.get("overlap") == OVERLAP
            and meta.get("scale") == SCALE
            and meta.get("output_sha256") == array_sha256(np.load(ref_npy, mmap_mode="r"))
        ):
            print(f"Reusing verified eager FP32 reference at {ref_npy}", flush=True)
            return ref_npy
    print("Generating fresh untouched eager CUDA FP32 128x128 reference...", flush=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--generate-reference", str(reference_dir)]
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "NVIDIA_TF32_OVERRIDE": "0"})
    return ref_npy


@torch.inference_mode()
def generate_reference_worker(reference_dir: Path) -> None:
    configure_runtime()
    reference_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(PAGE_PATH) as img:
        page = image_to_tensor(img)
    model, _ = load_checkpoint(CHECKPOINT)
    runner = TorchHostRunner(model, torch.float32)
    started = time.perf_counter()
    actual = infer_tensor(page, runner)
    elapsed = time.perf_counter() - started
    arr = actual.numpy()
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite eager FP32 reference output")
    ref_npy = reference_dir / "eager-fp32.npy"
    np.save(ref_npy, arr)
    write_json(
        reference_dir / "meta.json",
        {
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": digest(CHECKPOINT),
            "page": str(PAGE_PATH.relative_to(ROOT)),
            "page_sha256": digest(PAGE_PATH),
            "tile": TILE,
            "overlap": OVERLAP,
            "scale": SCALE,
            "output_shape": list(arr.shape),
            "output_sha256": array_sha256(arr),
            "elapsed_seconds": elapsed,
            "runtime": runtime_metadata(),
        },
    )
    print(f"Wrote eager FP32 reference to {ref_npy} ({elapsed:.3f} s)", flush=True)


def spawn_worker(directory: Path, name: str, mode: str, tag: str) -> dict[str, Any]:
    out_path = directory / mode / f"{tag}{name}.json"
    if out_path.is_file():
        return json.loads(out_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output",
        str(directory),
        "--worker",
        name,
        "--mode",
        mode,
        "--result",
        str(out_path),
    ]
    print(f"[{mode}] {tag}{name}", flush=True)
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "NVIDIA_TF32_OVERRIDE": "0"}
    )
    log_prefix = directory / "logs" / f"{mode}-{tag}{name}"
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".stdout.log").write_text(proc.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"Worker failed: {mode} {tag}{name}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Output directory for reports and raw arrays")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "output/gfisrv2-session/matched-128-ref",
        help="Directory holding the verified 128x128 eager FP32 reference",
    )
    parser.add_argument(
        "--runner",
        action="append",
        help="Runner spec: 'eager-fp32', 'eager-fp16', or 'name=precision:dll_path[|k=v,...]'",
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--skip-vram", action="store_true")
    parser.add_argument("--generate-reference", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("quality", "timing", "vram"), help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.generate_reference is not None:
        generate_reference_worker(args.generate_reference.resolve())
        return
    if args.worker is not None:
        run_worker(args)
        return

    if args.output is None or not args.runner:
        parser.error("--output and at least one --runner are required")
    specs = [parse_runner_spec(raw) for raw in args.runner]
    if len({s.name for s in specs}) != len(specs):
        parser.error("Runner names must be unique")

    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    reference_npy = ensure_reference(args.reference_dir.resolve())

    package_path = directory / "gfisrv2.ummodel"
    if not package_path.is_file():
        export_model(CHECKPOINT, package_path, kernel="gfisrv2")
    definition = {
        "git_revision": git_revision(),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": digest(CHECKPOINT),
        "page": str(PAGE_PATH.relative_to(ROOT)),
        "page_sha256": digest(PAGE_PATH),
        "tile": TILE,
        "overlap": OVERLAP,
        "scale": SCALE,
        "rounds": args.rounds,
        "package": {"path": str(package_path), "sha256": digest(package_path), "bytes": package_path.stat().st_size},
        "runners": {s.name: serialize_spec(s) for s in specs},
    }
    write_json(directory / "definition.json", definition)

    names = [s.name for s in specs]
    quality_reports: dict[str, Any] = {}
    for name in names:
        quality_reports[name] = spawn_worker(directory, name, "quality", "")

    rounds_data: list[dict[str, Any]] = []
    for r in range(args.rounds):
        order = names if r % 2 == 0 else list(reversed(names))
        row: dict[str, Any] = {"index": r, "order": order, "runners": {}}
        for name in order:
            res = spawn_worker(directory, name, "timing", f"round-{r}-")
            row["runners"][name] = res
            print(
                f"  round {r} {name}: {res['page_seconds']:.4f} s ({res['page_input_mpx_per_second']:.4f} MPx/s)",
                flush=True,
            )
        rounds_data.append(row)
        write_json(directory / "timing-report.json", {"rounds": rounds_data})

    vram_reports: dict[str, Any] = {}
    if not args.skip_vram:
        for name in names:
            vram_reports[name] = spawn_worker(directory, name, "vram", "")

    fidelity_scores = run_scoring(directory, reference_npy, names)

    summary: dict[str, Any] = {}
    for name in names:
        samples = [r["runners"][name]["page_seconds"] for r in rounds_data]
        med_s = statistics.median(samples)
        q = quality_reports[name]
        f = fidelity_scores[name]
        v = vram_reports.get(name)
        entry: dict[str, Any] = {
            "median_page_seconds": med_s,
            "min_page_seconds": min(samples),
            "max_page_seconds": max(samples),
            "samples_page_seconds": samples,
            "median_input_mpx_per_second": (900 * 1350) / med_s / 1e6,
            "average_batch1_device_tile_latency_ms": q["average_batch1_device_tile_latency_ms"],
            "median_batch1_device_tile_latency_ms": q["median_batch1_device_tile_latency_ms"],
            "average_batch1_host_tile_latency_ms": q["average_batch1_host_tile_latency_ms"],
            "median_batch1_host_tile_latency_ms": q["median_batch1_host_tile_latency_ms"],
            "psnr_db": f["psnr_db"],
            "max_abs": f["max_abs"],
            "ssimulacra2_display8": f["ssimulacra2_display8"],
            "ssimulacra2_rgb16": f["ssimulacra2_rgb16"],
            "graph_direct_bit_identical": q["graph_direct_bit_identical"],
            "output_sha256": q["output_sha256"],
        }
        if v is not None:
            entry["vram_steady_peak_above_idle_mib"] = v["steady_full_page"]["peak_above_pre_context_background_mib"]
            entry["vram_startup_peak_above_idle_mib"] = v["startup_compile_warmup"][
                "peak_above_pre_context_background_mib"
            ]
            entry["vram_warmed_device_delta_mib"] = v["warmed_device_vram_delta_mib"]
            if v["session_info"] is not None:
                entry["session_buffer_mib"] = {k: val / MIB for k, val in v["session_info"]["buffer_bytes"].items()}
        summary[name] = entry
    write_json(directory / "summary.json", {"definition": definition, "summary": summary})
    print("\n=== Matched 128x128 Full-Page Summary ===", flush=True)
    for name, s in summary.items():
        psnr = "inf" if s["psnr_db"] is None else f"{s['psnr_db']:.4f}"
        vram = (
            f"{s.get('vram_steady_peak_above_idle_mib', float('nan')):.2f} MiB"
            if "vram_steady_peak_above_idle_mib" in s
            else "skipped"
        )
        med_s = s["median_page_seconds"]
        min_s, max_s = s["min_page_seconds"], s["max_page_seconds"]
        mpx = s["median_input_mpx_per_second"]
        dev_ms = s["average_batch1_device_tile_latency_ms"]
        host_ms = s["average_batch1_host_tile_latency_ms"]
        ssim_g = s["ssimulacra2_display8"]
        print(
            f"{name:18s} | median {med_s:.4f} s ({min_s:.4f}-{max_s:.4f}) "
            f"| {mpx:.4f} MPx/s | dev {dev_ms:.3f} ms/tile (host {host_ms:.3f}) "
            f"| VRAM {vram} | PSNR {psnr} dB | gray SSIM2 {ssim_g:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
