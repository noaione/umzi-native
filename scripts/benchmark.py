"""Run the fixed GateRV3 baseline matrix: 128px tiles, batch 16, FP16 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model import load_model

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def tensor_image(image: Image.Image) -> torch.Tensor:
    a = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous()


def image_tensor(t: torch.Tensor) -> Image.Image:
    a = (t[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(
        np.uint8
    )
    return Image.fromarray(a, "RGB")


def patches(image: torch.Tensor, tile: int):
    _, _, h, w = image.shape
    coords = [(y, x) for y in range(0, h, tile) for x in range(0, w, tile)]
    chunks = []
    for y, x in coords:
        p = image[:, :, y : min(y + tile, h), x : min(x + tile, w)]
        p = F.pad(p, (0, tile - p.shape[-1], 0, tile - p.shape[-2]), mode="replicate")
        chunks.append((y, x, p))
    return chunks


@torch.inference_mode()
def run_tiled(run, image, tile, batch, device):
    h, w = image.shape[-2:]
    out = torch.empty((1, 3, h, w), dtype=torch.float32)
    target = getattr(run, "_orig_mod", run)
    input_dtype = (
        next(target.parameters()).dtype
        if isinstance(target, torch.nn.Module)
        else torch.float32
    )
    items = patches(image, tile)
    for i in range(0, len(items), batch):
        group = items[i : i + batch]
        x = torch.cat([v[2] for v in group]).to(device=device, dtype=input_dtype)
        y = run(x).float().cpu()
        for j, (top, left, _) in enumerate(group):
            hh, ww = min(tile, h - top), min(tile, w - left)
            out[:, :, top : top + hh, left : left + ww] = y[j : j + 1, :, :hh, :ww]
    return out


@torch.inference_mode()
def measure_single_tile_latencies(run, label, image, tile, device):
    """Time each tile as its own synchronized batch-1 invocation."""
    target = getattr(run, "_orig_mod", run)
    input_dtype = (
        next(target.parameters()).dtype
        if isinstance(target, torch.nn.Module)
        else torch.float32
    )
    samples = patches(image, tile)
    # Exclude first-call setup from per-tile samples.
    warmup = samples[0][2].to(device=device, dtype=input_dtype)
    run(warmup)
    torch.cuda.synchronize()
    timings = []
    for tile_index, (top, left, patch) in enumerate(samples):
        x = patch.to(device=device, dtype=input_dtype)
        torch.cuda.synchronize()
        started = time.perf_counter()
        run(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        timings.append(
            {
                "image": label,
                "tile_index": tile_index,
                "top": top,
                "left": left,
                "seconds": elapsed,
                "milliseconds": elapsed * 1000,
            }
        )
    return timings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/1x_umzi_digital_decompress_gaterv3_1.pth",
    )
    ap.add_argument("--input", type=Path, default=ROOT / "samples/input")
    ap.add_argument("--output", type=Path, default=ROOT / "output/baseline")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip-trt", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this baseline matrix")
    if args.tile < 16 or args.tile % 16 or args.batch < 1:
        raise SystemExit("tile must be a multiple of 16 and batch must be positive")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    paths = (
        sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if args.input.is_dir()
        else [args.input]
    )
    if not paths:
        raise SystemExit(f"No images found under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    model, config = load_model(args.model, device=device)
    print(
        f"GateRV3 {config.describe()} | {sum(p.numel() for p in model.parameters()):,} parameters",
        flush=True,
    )
    images = [(p.name, tensor_image(Image.open(p))) for p in paths]
    report: dict[str, Any] = {
        "config": config.describe(),
        "tile": args.tile,
        "batch": args.batch,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "checkpoint_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "variants": {},
        "quality_tool": r"C:\Tools\imaging\ssimulacra2.exe",
        "metric_definitions": {
            "per_tile_latency": "synchronized batch-1 wall time measured separately for every tile; startup warmup excluded",
            "average_per_tile_latency": "arithmetic mean of the individually measured batch-1 tile latencies",
            "batch16_seconds": "synchronized inference wall time for batch-16 tiled inference",
            "megapixels_per_second": "source image pixels divided by synchronized inference wall time",
            "psnr_db": "raw float model output versus the matching Torch CUDA FP32 output, peak=1, before PNG quantization",
        },
    }
    references: list[torch.Tensor] | None = None

    def evaluate(name, runner, directory, single_tile_runner=None):
        nonlocal references
        directory.mkdir(parents=True, exist_ok=True)
        results: list[torch.Tensor] = []
        image_stats = []
        individual_tile_times = []
        total_tiles = 0
        total_pixels = 0
        for index, (label, img) in enumerate(images):
            tile_count = len(patches(img, args.tile))
            torch.cuda.synchronize()
            image_start = time.perf_counter()
            pred = run_tiled(runner, img, args.tile, args.batch, device)
            torch.cuda.synchronize()
            image_seconds = time.perf_counter() - image_start
            image_tensor(pred).save(directory / (Path(label).stem + ".png"))
            results.append(pred)
            tile_total = tile_count
            pixel_count = img.shape[-2] * img.shape[-1]
            row: dict[str, Any] = {
                "image": label,
                "tiles": tile_total,
                "batch16_inference_seconds": image_seconds,
                "megapixels_per_second": pixel_count / image_seconds / 1e6,
            }
            if references is not None:
                delta = pred.double() - references[index].double()
                mse = float(delta.square().mean())
                row["psnr_db"] = None if mse == 0 else float(10 * np.log10(1 / mse))
            image_stats.append(row)
            tile_timings = measure_single_tile_latencies(
                single_tile_runner or runner, label, img, args.tile, device
            )
            row["individual_tile_latency_seconds"] = [
                item["seconds"] for item in tile_timings
            ]
            row["average_individual_tile_latency_seconds"] = float(
                np.mean(row["individual_tile_latency_seconds"])
            )
            row["average_individual_tile_latency_milliseconds"] = (
                row["average_individual_tile_latency_seconds"] * 1000
            )
            individual_tile_times.extend(tile_timings)
            print(
                f"{name} {label}: batch-{args.batch} {image_seconds:.2f}s; "
                f"measured {len(tile_timings)} individual tiles",
                flush=True,
            )
            total_tiles += tile_total
            total_pixels += pixel_count
        torch.cuda.synchronize()
        overall_seconds = sum(row["batch16_inference_seconds"] for row in image_stats)
        report["variants"][name] = {
            "batch16_inference_seconds": overall_seconds,
            "images": len(images),
            "tiles": total_tiles,
            "batch16_megapixels_per_second": total_pixels / overall_seconds / 1e6,
            "megapixels_per_second": total_pixels / overall_seconds / 1e6,
            "individual_tile_latency": {
                "measurement_batch_size": 1,
                "timing_scope": "one synchronized forward call for each tile, including host/device transfer",
                "measured_tile_count": len(individual_tile_times),
                "average_seconds": float(
                    np.mean([item["seconds"] for item in individual_tile_times])
                ),
                "average_milliseconds": float(
                    np.mean([item["milliseconds"] for item in individual_tile_times])
                ),
                "per_tile": individual_tile_times,
            },
            "outputs": str(directory),
            "per_image": image_stats,
        }
        if references is None:
            references = results
            report["variants"][name]["reference"] = "Torch CUDA FP32"
        else:
            total_squared_error = 0.0
            total_values = 0
            for result, reference in zip(results, references, strict=True):
                delta = result.double() - reference.double()
                total_squared_error += float(delta.square().sum())
                total_values += delta.numel()
            mse = total_squared_error / total_values
            report["variants"][name]["psnr_db"] = (
                None if mse == 0 else float(10 * np.log10(1 / mse))
            )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"{name}: batch-{args.batch} {overall_seconds:.2f}s, "
            f"mean single-tile latency "
            f"{report['variants'][name]['individual_tile_latency']['average_milliseconds']:.3f}ms",
            flush=True,
        )
        return results

    evaluate("torch_cuda_fp32_reference", model, args.output / "torch-fp32")
    model16, _ = load_model(args.model, device=device, dtype=torch.float16)
    evaluate("torch_cuda_fp16", model16, args.output / "torch-fp16")
    compiled = torch.compile(model16, mode="reduce-overhead", dynamic=False)
    # Explicitly warm up the exact batch/tile signature before timed execution.
    with torch.inference_mode():
        compiled(
            torch.zeros(
                args.batch, 3, args.tile, args.tile, device=device, dtype=torch.float16
            )
        )
    torch.cuda.synchronize()
    evaluate("torch_inductor_cuda_fp16", compiled, args.output / "inductor-fp16")

    if not args.skip_trt:
        onnx_path = args.output / f"gaterv3-fp16-t{args.tile}-b{args.batch}.onnx"
        if not onnx_path.exists():
            dummy = torch.zeros(
                args.batch, 3, args.tile, args.tile, device=device, dtype=torch.float16
            )
            torch.onnx.export(
                model16,
                (dummy,),
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                dynamo=False,
            )
        import onnxruntime as ort
        import onnxruntime_ep_nv_tensorrt_rtx as rtx

        ort.disable_telemetry_events()
        ort.register_execution_provider_library(
            rtx.get_ep_name(), rtx.get_library_path()
        )
        opt = ort.SessionOptions()
        opt.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        opt.add_provider_for_devices(
            [d for d in ort.get_ep_devices() if d.ep_name == rtx.get_ep_name()], {}
        )
        sess = ort.InferenceSession(str(onnx_path), sess_options=opt, enable_fallback=0)
        if rtx.get_ep_name() not in sess.get_providers():
            raise RuntimeError(f"TensorRT-RTX did not activate: {sess.get_providers()}")
        input_name = sess.get_inputs()[0].name

        single_onnx_path = args.output / f"gaterv3-fp16-t{args.tile}-b1.onnx"
        if not single_onnx_path.exists():
            single_dummy = torch.zeros(
                1, 3, args.tile, args.tile, device=device, dtype=torch.float16
            )
            torch.onnx.export(
                model16,
                (single_dummy,),
                str(single_onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                dynamo=False,
            )
        single_opt = ort.SessionOptions()
        single_opt.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        single_opt.add_provider_for_devices(
            [d for d in ort.get_ep_devices() if d.ep_name == rtx.get_ep_name()], {}
        )
        single_sess = ort.InferenceSession(
            str(single_onnx_path), sess_options=single_opt, enable_fallback=0
        )
        if rtx.get_ep_name() not in single_sess.get_providers():
            raise RuntimeError(
                f"TensorRT-RTX batch-1 session did not activate: {single_sess.get_providers()}"
            )
        single_input_name = single_sess.get_inputs()[0].name

        def make_ort_runner(session, name):
            def run(x):
                arr = x.detach().cpu().numpy().astype(np.float16, copy=False)
                return torch.from_numpy(session.run(None, {name: arr})[0]).to(
                    device=device, dtype=torch.float32
                )

            return run

        ort_run = make_ort_runner(sess, input_name)
        single_ort_run = make_ort_runner(single_sess, single_input_name)

        evaluate(
            "onnx_tensorrt_rtx_fp16",
            ort_run,
            args.output / "trt-rtx-fp16",
            single_tile_runner=single_ort_run,
        )
        report["variants"]["onnx_tensorrt_rtx_fp16"]["providers"] = sess.get_providers()
        report["variants"]["onnx_tensorrt_rtx_fp16"]["batch1_providers"] = (
            single_sess.get_providers()
        )
        report["variants"]["onnx_tensorrt_rtx_fp16"]["onnx"] = str(onnx_path)
        report["variants"]["onnx_tensorrt_rtx_fp16"]["batch1_onnx"] = str(
            single_onnx_path
        )

    report["quality_comparison"] = (
        "ssimulacra2 pairwise against per-image Torch CUDA FP32 output"
    )
    quality_tool = Path(r"C:\Tools\imaging\ssimulacra2.exe")
    report["quality_scores"] = {}
    if quality_tool.is_file():
        for variant, output_dir in (
            ("torch_cuda_fp16", args.output / "torch-fp16"),
            ("torch_inductor_cuda_fp16", args.output / "inductor-fp16"),
            ("onnx_tensorrt_rtx_fp16", args.output / "trt-rtx-fp16"),
        ):
            if not output_dir.is_dir():
                continue
            scores = []
            for label, _ in images:
                candidate = output_dir / (Path(label).stem + ".png")
                reference = args.output / "torch-fp32" / candidate.name
                if not candidate.is_file() or not reference.is_file():
                    continue
                result = subprocess.run(
                    [str(quality_tool), str(reference), str(candidate)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                values = re.findall(r"-?\d+(?:\.\d+)?", result.stdout)
                if not values:
                    raise RuntimeError(
                        f"Could not parse SSIMULACRA2 output: {result.stdout!r}"
                    )
                scores.append({"image": candidate.name, "score": float(values[-1])})
            if scores:
                report["quality_scores"][variant] = {
                    "per_image": scores,
                    "mean": sum(item["score"] for item in scores) / len(scores),
                    "minimum": min(item["score"] for item in scores),
                }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
