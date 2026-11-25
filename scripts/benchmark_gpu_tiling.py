"""Compare fixed CUDA page tiling with nmanga's fixed CPU tiler on the same native model."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from benchmark import IMAGE_EXTS, ROOT, load_model
from benchmark_architectures import comparison, digest, ssimulacra2
from benchmark_manga import CaptureBeforeClip, TorchReference
from PIL import Image
from reference.native_session import DeviceSession
from reference.nvml import Nvml, NvmlSampler

from umzi_native import InferenceSession, SessionOptions
from umzi_native.exporter._loader import load_checkpoint

MODELS = {
    "gaterv3": ("1x_umzi_digital_decompress_gaterv3_1.pth", "input", 128, 16, 16),
    "smosr": ("2x_enhancr_da_smosr_v1.safetensors", "2x-test", 256, 1, 32),
}


def cpu_page(
    backend: ModuleType, image: Image.Image, session: Any, tile: int, batch: int, overlap: int, background: str
):
    return backend.denoise_single_image_with_overlap(
        image,
        session,
        tile_size=tile,
        batch_size=batch,
        tile_overlap=overlap,
        background=background,
        pad_final_batch=True,
        use_fp32=session.get_inputs()[0].type == "tensor(float)",
    )


def cpu_raw(
    backend: ModuleType,
    tiler: ModuleType,
    image: Image.Image,
    session: Any,
    tile: int,
    batch: int,
    overlap: int,
    background: str,
) -> tuple[np.ndarray, np.ndarray]:
    if overlap:
        capture = CaptureBeforeClip()
        with patch.object(backend, "get_numpy", return_value=capture):
            with cpu_page(backend, image, session, tile, batch, overlap, background) as rendered:
                assert capture.image is not None
                return np.moveaxis(capture.image, 0, -1), np.array(rendered)

    # The CPU no-overlap path converts each valid tile straight into the byte canvas.
    boxes = iter(tiler.plan_tiles(*image.size, tile_size=tile))
    scale = session.scale_factor
    raw = np.empty((image.height * scale, image.width * scale, 3), np.float32)
    quantize = backend._quantize_tile_to_uint8

    def capture_tile(output: np.ndarray) -> np.ndarray:
        box = next(boxes)
        y, x = box.dst_y * scale, box.dst_x * scale
        _, h, w = output.shape
        raw[y : y + h, x : x + w] = np.moveaxis(output, 0, -1)
        return quantize(output)

    with patch.object(backend, "_quantize_tile_to_uint8", side_effect=capture_tile):
        with cpu_page(backend, image, session, tile, batch, overlap, background) as rendered:
            assert next(boxes, None) is None
            return raw, np.array(rendered)


def smoke(backend: ModuleType, tiler: ModuleType, session: InferenceSession, tile: int, batch: int) -> list[dict]:
    rows = []
    for height, width, overlap, background in (
        (1, 1, 0, "black"),
        (1, 17, tile // 4, "white"),
        (tile + 1, tile + 3, 0, "white"),
        (tile + 1, tile + 5, tile // 2 + 1, "white"),
        (tile * 2 + 7, tile * 3 + 1, tile // 4, "white"),
    ):
        y, x, c = np.ogrid[:height, :width, :3]
        array = ((y * 71 + x * 29 + c * 11) % 256).astype(np.uint8)
        with Image.fromarray(array) as image:
            raw, quantized = cpu_raw(backend, tiler, image, session, tile, batch, overlap, background)
        result = session.run_tiled(
            array, tile_size=tile, batch_size=batch, overlap=overlap, background=background, output_dtype=np.float32
        )
        rendered = session.run_tiled(array, tile_size=tile, batch_size=batch, overlap=overlap, background=background)
        np.testing.assert_allclose(result, raw, rtol=1e-6, atol=1e-6)
        assert np.max(np.abs(rendered.astype(np.int16) - quantized)) <= int(overlap != 0)
        rows.append({"size": [width, height], "overlap": overlap, "max_abs": float(np.max(np.abs(result - raw)))})
    for limit, blend, fill in ((batch, -1, "white"), (batch, tile, "white"), (0, 0, "other"), (1, 0, "other")):
        try:
            session.run_tiled(array, tile_size=tile, batch_size=limit, overlap=blend, background=fill)
        except ValueError:
            continue
        raise AssertionError(f"Expected validation error for {limit}, {blend}, {fill}")
    return rows


def model_calls(
    backend: ModuleType,
    tiler: ModuleType,
    session: InferenceSession,
    image: Image.Image,
    tile: int,
    batch: int,
    overlap: int,
) -> list[dict]:
    dtype = np.float16 if session.get_inputs()[0].type == "tensor(float16)" else np.float32
    lookup = (np.arange(256) / 255).astype(dtype)
    source = np.asarray(image).transpose(2, 0, 1)
    boxes = tiler.plan_tiles(*image.size, tile_size=tile, tile_overlap=overlap)
    device = DeviceSession(session, reuse_output=True)
    result = []
    for start in range(0, len(boxes), batch):
        group = boxes[start : start + batch]
        real = len(group)
        sample = backend._build_tile_batch(
            source,
            group + [group[-1]] * (batch - real),
            run_h=tile,
            run_w=tile,
            model_dtype=dtype,
            lookup=lookup,
            pad_value=None if overlap else 255,
        )
        sample_gpu = torch.from_numpy(sample).cuda(0)
        torch.cuda.synchronize(0)
        tick = time.perf_counter()
        device(sample_gpu)
        torch.cuda.synchronize(0)
        result.append({
            "tile_index": start,
            "left": group[0].dst_x,
            "top": group[0].dst_y,
            "real_tiles": real,
            "submitted_tiles": batch,
            "elapsed_ms": (time.perf_counter() - tick) * 1000,
        })
    return result


def render_paths(directory: Path, images: dict[str, np.ndarray]) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{name}.png" for name in images}
    for name, pixels in images.items():
        with Image.fromarray(pixels) as image:
            image.save(paths[name])
    return paths


def quality(
    backend: ModuleType,
    tiler: ModuleType,
    image: Image.Image,
    reference: TorchReference,
    session: InferenceSession,
    tile: int,
    batch: int,
    overlap: int,
    directory: Path,
) -> dict:
    src = np.asarray(image)
    gpu_raw = session.run_tiled(src, tile_size=tile, batch_size=batch, overlap=overlap, output_dtype=np.float32)
    gpu_image = session.run_tiled(src, tile_size=tile, batch_size=batch, overlap=overlap)
    cpu_values, cpu_image = cpu_raw(backend, tiler, image, session, tile, batch, overlap, "white")
    eager_values, eager_image = cpu_raw(backend, tiler, image, reference, tile, batch, overlap, "white")
    parity = comparison(np.moveaxis(gpu_raw, -1, 0), np.moveaxis(cpu_values, -1, 0))
    assert parity["finite"] and parity["max_abs"] <= 2e-6, parity
    max_byte_error = int(np.max(np.abs(gpu_image.astype(np.int16) - cpu_image)))
    assert max_byte_error <= int(overlap != 0), max_byte_error
    gpu_fp32 = comparison(np.moveaxis(gpu_raw, -1, 0), np.moveaxis(eager_values, -1, 0))
    cpu_fp32 = comparison(np.moveaxis(cpu_values, -1, 0), np.moveaxis(eager_values, -1, 0))
    assert gpu_fp32["finite"] and cpu_fp32["finite"]
    images = {"torch-fp32": eager_image, "cpu-tiles": cpu_image, "gpu-tiles": gpu_image}
    paths = render_paths(directory, images)
    if overlap:
        scale = session.scale_factor
        seam = (tile - overlap) * scale
        render_paths(
            directory / "seam",
            {name: pixels[seam - 32 : seam + 96, seam - 32 : seam + 96] for name, pixels in images.items()},
        )
    return {
        "gpu_vs_cpu": parity,
        "max_byte_error": max_byte_error,
        "gpu_vs_fp32": gpu_fp32,
        "cpu_vs_fp32": cpu_fp32,
        "ssimulacra2_gpu": ssimulacra2(paths["torch-fp32"], paths["gpu-tiles"]),
        "ssimulacra2_cpu": ssimulacra2(paths["torch-fp32"], paths["cpu-tiles"]),
    }


def timed_pages(
    backend: ModuleType,
    image: Image.Image,
    session: InferenceSession,
    nvml: Nvml,
    tile: int,
    batch: int,
    overlap: int,
    rounds: int,
    background_mib: float,
) -> dict:
    source = np.asarray(image)
    phases = {
        "cpu": lambda: cpu_page(backend, image, session, tile, batch, overlap, "white").close(),
        "gpu": lambda: Image.fromarray(
            session.run_tiled(source, tile_size=tile, batch_size=batch, overlap=overlap)
        ).close(),
    }
    samples: dict[str, list[dict]] = {name: [] for name in phases}
    for index in range(rounds):
        for name in phases if index % 2 == 0 else reversed(phases):
            with NvmlSampler(nvml) as sampler:
                start = time.perf_counter()
                phases[name]()
                elapsed = (time.perf_counter() - start) * 1000
            samples[name].append({"ms": elapsed, **sampler.report(background_mib)})
    return {
        name: {
            "median_ms": statistics.median(row["ms"] for row in rows),
            "peak_mib": max(row["device_peak_mib"] for row in rows),
            "rounds": rows,
        }
        for name, rows in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=MODELS)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manga-repo", type=Path, default=ROOT.parent / "nao-manga-rls")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    sys.path.insert(0, str(args.manga_repo.resolve()))
    backend = importlib.import_module("nmanga.ml.onnx_backend")
    tiler = importlib.import_module("nmanga.ml.tiler")
    checkpoint, samples, tile, batch, test_overlap = MODELS[args.family]
    checkpoint_path = ROOT / "models" / checkpoint
    device_id = 0
    nvml = Nvml()
    try:
        background_mib = nvml.read()["used_mib"]
        torch.cuda.set_device(device_id)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(torch.cuda.get_device_name(device_id), args.family, flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        options = SessionOptions(
            precision="fp16",
            device_id=device_id,
            dll_path=args.library,
            cache_dir=args.output.parent / "cache",
            enable_cuda_graph=not args.no_cuda_graph,
        )
        report: dict[str, Any] = {
            "gpu": torch.cuda.get_device_name(device_id),
            "background_mib": background_mib,
            "tile": tile,
            "batch": batch,
            "model_sha256": digest(args.model),
            "checkpoint_sha256": digest(checkpoint_path),
            "dll_sha256": digest(args.library),
            "cases": {},
        }
        with InferenceSession(args.model, options) as session:
            report["smoke"] = smoke(backend, tiler, session, tile, batch)
            print("GPU/CPU smoke cases passed", flush=True)
            if not args.smoke_only:
                model = (
                    load_model(checkpoint_path, device="cuda:0")[0]
                    if args.family == "gaterv3"
                    else (load_checkpoint(checkpoint_path)[0].eval().cuda(device_id))
                )
                ref = TorchReference(model, torch.float32, batch, tile)
                ref.scale_factor = session.scale_factor
                for overlap in (0, test_overlap):
                    rows = {}
                    for path in sorted((ROOT / "samples" / samples).iterdir()):
                        if path.suffix.lower() not in IMAGE_EXTS:
                            continue
                        with Image.open(path) as file, file.convert("RGB") as image:
                            timing = timed_pages(
                                backend, image, session, nvml, tile, batch, overlap, args.rounds, background_mib
                            )
                            calls = model_calls(backend, tiler, session, image, tile, batch, overlap)
                            fidelity = quality(
                                backend,
                                tiler,
                                image,
                                ref,
                                session,
                                tile,
                                batch,
                                overlap,
                                args.output / f"overlap-{overlap}" / path.stem,
                            )
                        rows[path.name] = {
                            "page": timing,
                            "fidelity": fidelity,
                            "batch_calls": calls,
                            "input_sha256": digest(path),
                        }
                        print(
                            f"overlap {overlap}, {path.name}: CPU {timing['cpu']['median_ms']:.1f} ms, "
                            f"GPU {timing['gpu']['median_ms']:.1f} ms",
                            flush=True,
                        )
                    report["cases"][str(overlap)] = rows
                del model, ref
        if not args.smoke_only:
            torch.cuda.empty_cache()
            with InferenceSession(args.model, options) as session:
                session.prepare((1, 3, tile, tile))
                for overlap, rows in report["cases"].items():
                    for name, row in rows.items():
                        with Image.open(ROOT / "samples" / samples / name) as file, file.convert("RGB") as image:
                            calls = (
                                row["batch_calls"]
                                if batch == 1
                                else (model_calls(backend, tiler, session, image, tile, 1, int(overlap)))
                            )
                        row["batch1_samples"] = [{"image": name, **call} for call in calls]
                        row["batch1_mean_ms"] = statistics.mean(call["elapsed_ms"] for call in calls)
                        row["batch_inference_ms"] = sum(call["elapsed_ms"] for call in row["batch_calls"])
                        row["useful_tiles_per_second"] = (
                            1000 * sum(call["real_tiles"] for call in row["batch_calls"]) / (row["batch_inference_ms"])
                        )
                    total_values = sum(row["fidelity"]["gpu_vs_fp32"]["values"] for row in rows.values())
                    mse = (
                        sum(
                            row["fidelity"]["gpu_vs_fp32"]["mse"] * row["fidelity"]["gpu_vs_fp32"]["values"]
                            for row in rows.values()
                        )
                        / total_values
                    )
                    report.setdefault("summary", {})[overlap] = {
                        "gpu_psnr_db": -10 * math.log10(mse) if mse else None,
                        "gpu_ssimulacra2_mean": statistics.mean(
                            row["fidelity"]["ssimulacra2_gpu"] for row in rows.values()
                        ),
                        "batch_ms": sum(row["batch_inference_ms"] for row in rows.values()),
                        "batch1_mean_ms": statistics.mean(
                            call["elapsed_ms"] for row in rows.values() for call in row["batch1_samples"]
                        ),
                        **{
                            f"{name}_page_ms": sum(row["page"][name]["median_ms"] for row in rows.values())
                            for name in ("cpu", "gpu")
                        },
                    }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        nvml.close()


if __name__ == "__main__":
    main()
