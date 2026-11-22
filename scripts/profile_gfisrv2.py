"""Profile GFISRv2 native sessions (CUPTI kernel breakdown + host page decomposition)."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from benchmark_gfisrv2_page import OVERLAP, PAGE_PATH, SCALE, TILE, NativeHostRunner, image_to_tensor, infer_tensor
from PIL import Image
from reference.native_session import DeviceSession
from torch.profiler import ProfilerActivity, profile

from umzi_native import InferenceSession, SessionOptions


def classify_kernel(name: str) -> str:
    low = name.lower()
    if "fft" in low or "cufft" in low:
        return "cuFFT"
    if any(
        tok in low
        for tok in (
            "cudnn",
            "ampere_h16816",
            "ampere_fp16",
            "volta_",
            "turing_",
            "convolve",
            "winograd",
            "implicit_convolve",
            "nhwc",
            "nchw",
            "cutlass",
            "xmma",
        )
    ):
        if not any(
            custom in low
            for custom in (
                "rms_norm",
                "depthwise",
                "copy_channels",
                "pack_fft",
                "unpack_fft",
                "scale_fft",
                "add_bias",
                "gate",
                "residual",
                "apply_silu",
                "add_features",
                "cast_to_",
                "add_half_residual",
                "shuffle_rgb",
                "expand_gate",
                "project_",
                "pack_rgb",
                "input_bias",
                "output_",
                "fourier_",
            )
        ):
            return "cuDNN/GEMM"
    if "memcpy" in low or "memset" in low:
        return "memcpy/memset"
    return "custom_kernel"


@torch.inference_mode()
def profile_device(
    model: Path,
    library: Path,
    precision: str,
    batch: int,
    tile: int,
    repeats: int,
    kernel_options: dict[str, int],
) -> dict[str, Any]:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    shape = (batch, 3, tile, tile)
    options = SessionOptions(
        precision=precision,
        dll_path=library,
        enable_cuda_graph=False,
        kernel_options=kernel_options,
    )
    with InferenceSession(model, options) as session:
        session.prepare(shape)
        info = session.get_session_info()
        runner = DeviceSession(session)
        sample = torch.rand(shape, device="cuda", dtype=dtype)
        for _ in range(3):
            runner(sample)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(repeats):
                runner(sample)
            torch.cuda.synchronize()

    events = prof.key_averages()
    by_kernel: list[dict[str, Any]] = []
    by_category: dict[str, float] = defaultdict(float)
    total_cuda_us = 0.0
    for evt in events:
        cuda_us = float(getattr(evt, "self_device_time_total", 0.0) or getattr(evt, "self_cuda_time_total", 0.0))
        if cuda_us <= 0.0:
            continue
        cat = classify_kernel(evt.key)
        by_category[cat] += cuda_us
        total_cuda_us += cuda_us
        by_kernel.append({
            "name": evt.key,
            "category": cat,
            "calls_per_forward": evt.count / repeats,
            "cuda_ms_per_forward": cuda_us / repeats / 1000.0,
        })
    by_kernel.sort(key=lambda row: row["cuda_ms_per_forward"], reverse=True)
    categories = {
        k: {"ms_per_forward": v / repeats / 1000.0, "share_pct": (100.0 * v / total_cuda_us) if total_cuda_us else 0.0}
        for k, v in sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)
    }
    return {
        "precision": precision,
        "shape": list(shape),
        "repeats": repeats,
        "total_kernel_ms_per_forward": total_cuda_us / repeats / 1000.0,
        "categories": categories,
        "top_kernels": by_kernel[:30],
        "session_info": info,
    }


@torch.inference_mode()
def profile_host_page(model: Path, library: Path, precision: str, kernel_options: dict[str, int]) -> dict[str, Any]:
    with Image.open(PAGE_PATH) as img:
        page = image_to_tensor(img)
    options = SessionOptions(
        precision=precision,
        dll_path=library,
        enable_cuda_graph=True,
        kernel_options=kernel_options,
    )
    with InferenceSession(model, options) as session:
        session.prepare((1, 3, TILE, TILE))
        host_runner = NativeHostRunner(session)
        _ = infer_tensor(page, host_runner, scale=SCALE, tile=TILE, overlap=OVERLAP)
        torch.cuda.synchronize()
        runner_seconds = 0.0
        calls = 0

        def timed_runner(tile_tensor: torch.Tensor) -> torch.Tensor:
            nonlocal runner_seconds, calls
            t0 = time.perf_counter()
            out = host_runner(tile_tensor)
            runner_seconds += time.perf_counter() - t0
            calls += 1
            return out

        t_page = time.perf_counter()
        _ = infer_tensor(page, timed_runner, scale=SCALE, tile=TILE, overlap=OVERLAP)
        total_seconds = time.perf_counter() - t_page
    return {
        "total_page_seconds": total_seconds,
        "runner_calls_seconds": runner_seconds,
        "cpu_tiling_stitching_seconds": total_seconds - runner_seconds,
        "tile_calls": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precisions", nargs="+", default=["fp16", "mixed"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--host-page", action="store_true")
    parser.add_argument("--option", action="append", default=[], help="key=int kernel option")
    args = parser.parse_args()

    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kernel_options = {k: int(v) for item in args.option for k, v in (item.split("=", 1),)}

    report: dict[str, Any] = {"library": str(args.library), "batch": args.batch, "tile": args.tile, "results": {}}
    for precision in args.precisions:
        print(f"=== Profiling {precision} batch={args.batch} tile={args.tile} ===", flush=True)
        entry = profile_device(args.model, args.library, precision, args.batch, args.tile, args.repeats, kernel_options)
        if args.host_page and args.batch == 1 and args.tile == 128:
            entry["host_page"] = profile_host_page(args.model, args.library, precision, kernel_options)
        report["results"][precision] = entry
        print(f"Total GPU kernel time: {entry['total_kernel_ms_per_forward']:.3f} ms/forward", flush=True)
        for cat, stats in entry["categories"].items():
            print(f"  {cat:15s}: {stats['ms_per_forward']:7.3f} ms ({stats['share_pct']:5.1f}%)", flush=True)
        print("  Top 12 kernels:", flush=True)
        for row in entry["top_kernels"][:12]:
            print(
                f"    {row['cuda_ms_per_forward']:7.3f} ms ({row['calls_per_forward']:5.1f}x) "
                f"[{row['category']:13s}] {row['name'][:90]}",
                flush=True,
            )
        if "host_page" in entry:
            hp = entry["host_page"]
            print(
                f"  Host page: total {hp['total_page_seconds']:.4f} s, runner {hp['runner_calls_seconds']:.4f} s, "
                f"CPU stitch/tile {hp['cpu_tiling_stitching_seconds']:.4f} s ({hp['tile_calls']} tiles)",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
