"""Benchmark GateRV3 through the actual nao-manga-rls PIL page tiler."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict
from unittest.mock import patch

import numpy as np
import torch
from benchmark import IMAGE_EXTS, ROOT, load_model
from benchmark_native import CHECKPOINT, METRIC
from PIL import Image
from rtx_session import create_rtx_session

import umzi_native

VARIANTS = ("native-fp16", "native-fp16-fixed", "native-mixed", "trt-rtx")
SETUP_PHASES = ("shape_allocation", "plans", "allocation_warmup", "capture")


class ObservedSession:
    """Track the host calls the application makes without changing the tiler."""

    scale_factor = 1
    context = 0

    def __init__(self, session: Any, fixed_batch: int | None = None, native: bool = False):
        self.session = session
        spec = session.get_inputs()[0]
        self.input_spec = SimpleNamespace(name=spec.name, type=spec.type, shape=list(spec.shape))
        if fixed_batch is not None:
            self.input_spec.shape[0] = fixed_batch
        self.native = native
        self.last_shape = None
        self.rows: list[dict[str, Any]] = []
        self.observer_ms = 0.0

    def get_inputs(self):
        return [self.input_spec]

    def get_outputs(self):
        return self.session.get_outputs()

    def run(self, names, feeds):
        sample = feeds[self.input_spec.name]
        started = time.perf_counter()
        result = self.session.run(names, feeds)
        finished = time.perf_counter()
        if result[0].shape != sample.shape:
            raise RuntimeError(f"Unexpected output {result[0].shape} for {sample.shape}")
        row: dict[str, Any] = {
            "submitted_tiles": sample.shape[0],
            "run_ms": (finished - started) * 1000,
            "input_c_contiguous": bool(sample.flags.c_contiguous),
        }
        if self.native and self.last_shape != sample.shape:
            info = self.session.get_session_info()
            row["native_setup_stages_ms"] = {phase: info["stages_ms"][phase] for phase in SETUP_PHASES}
            row["native_setup_ms"] = sum(row["native_setup_stages_ms"].values())
            row["cache_status"] = info["cache_status"]
        self.last_shape = sample.shape
        self.rows.append(row)
        self.observer_ms += (time.perf_counter() - finished) * 1000
        return result


class TorchReference:
    scale_factor = 1
    context = 0

    def __init__(self, model, dtype: torch.dtype, batch: int, tile: int):
        self.model = model
        self.dtype = dtype
        input_type = "tensor(float16)" if dtype == torch.float16 else "tensor(float)"
        self.input_spec = SimpleNamespace(name="input", shape=[batch, 3, tile, tile], type=input_type)

    def get_inputs(self):
        return [self.input_spec]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, _names, feeds):
        x = torch.from_numpy(feeds["input"]).to(device="cuda", dtype=self.dtype)
        with torch.inference_mode():
            return [self.model(x).cpu().numpy()]


class CaptureBeforeClip:
    """Capture the PIL tiler's FP32 blended canvas before clipping and PNG quantization."""

    image: np.ndarray | None = None

    def __getattr__(self, name):
        return getattr(np, name)

    def clip(self, array, *args, **kwargs):
        self.image = array.copy()
        return np.clip(array, *args, **kwargs)


class Fidelity(TypedDict):
    mse: float
    psnr_db: float | None
    ssimulacra2: float


def export_missing_onnx(directory: Path, batches: list[int], tile: int) -> None:
    paths = {batch: directory / f"gaterv3-fp16-t{tile}-b{batch}.onnx" for batch in batches}
    if all(path.is_file() for path in paths.values()):
        return
    directory.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(CHECKPOINT, device="cuda")
    model.half()
    with tempfile.TemporaryDirectory(dir=directory) as staging, torch.inference_mode():
        for batch, path in paths.items():
            if path.is_file():
                continue
            print(f"Exporting {path}", flush=True)
            sample = torch.zeros((batch, 3, tile, tile), device="cuda", dtype=torch.float16)
            temp = Path(staging) / path.name
            torch.onnx.export(
                model,
                (sample,),
                temp,
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                dynamo=False,
            )
            temp.replace(path)


@contextmanager
def open_session(args, variant: str, batch: int) -> Generator[tuple[ObservedSession, dict]]:
    if variant == "trt-rtx":
        onnx = args.onnx_dir / f"gaterv3-fp16-t{args.tile}-b{batch}.onnx"
        ort, config = create_rtx_session(onnx, batch, args.tile, args.rtx_cache)
        try:
            yield ObservedSession(ort), {"providers": ort.get_providers(), "provider_options": config}
        finally:
            del ort
    else:
        precision = "mixed" if variant == "native-mixed" else "fp16"
        options = umzi_native.SessionOptions(
            precision=precision,
            dll_path=args.library,
            cache_dir=args.native_cache,
        )
        fixed = batch if variant.endswith("-fixed") else None
        with umzi_native.InferenceSession(args.model, options) as session:
            yield ObservedSession(session, fixed, native=True), {"precision": precision}


def page(module, image, session, tile, batch, overlap):
    return module.denoise_single_image_with_overlap(
        image,
        session,
        batch_size=batch,
        tile_size=tile,
        tile_overlap=overlap,
        background="black",
        contrast_stretch=False,
        tta=False,
    )


def tile_grid(image, tile: int, overlap: int) -> tuple[int, int]:
    width, height = image.size
    step = tile - overlap
    cols, rows = (max(1, math.ceil((length - overlap) / step)) for length in (width, height))
    return cols, rows


def measure_page(module, image, name, observed, tile, batch, overlap, round_index):
    width, height = image.size
    step = tile - overlap
    cols, rows = tile_grid(image, tile, overlap)
    total = cols * rows
    observed.rows = []
    observed.observer_ms = 0.0
    started = time.perf_counter()
    rendered = page(module, image, observed, tile, batch, overlap)
    elapsed = (time.perf_counter() - started) * 1000 - observed.observer_ms
    with rendered:
        if rendered.size != image.size:
            raise RuntimeError(f"{name}: incorrect output size {rendered.size}")
    fixed = observed.input_spec.shape[0]
    for start, call in zip(range(0, total, batch), observed.rows, strict=True):
        real = min(batch, total - start)
        expected = fixed if isinstance(fixed, int) else real
        if call["submitted_tiles"] != expected:
            raise RuntimeError(f"{name}: submitted {call['submitted_tiles']} tiles, expected {expected}")
        call.update(first_tile=start, real_tiles=real, image=name)
        if batch == 1:
            y, x = divmod(start, cols)
            call.update(tile_index=start, top=y * step, left=x * step)
    inference = sum(call["run_ms"] for call in observed.rows)
    return {
        "image": name,
        "round": round_index,
        "size": [width, height],
        "real_tiles": total,
        "submitted_tiles": sum(call["submitted_tiles"] for call in observed.rows),
        "page_ms": elapsed,
        "inference_ms": inference,
        "other_ms": elapsed - inference,
        "native_setup_ms": sum(call.get("native_setup_ms", 0) for call in observed.rows),
        "calls": observed.rows,
    }


def render_raw(module, args, image, name, session, batch, overlap, variant):
    capture = CaptureBeforeClip()
    with patch.object(module, "get_numpy", return_value=capture):
        rendered = page(module, image, session, args.tile, batch, overlap)
    path = args.output / "renders" / f"overlap{overlap}" / variant / Path(name).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rendered:
        rendered.save(path)
    if capture.image is None:
        raise RuntimeError("The manga tiler did not expose a canvas before clipping")
    raw = capture.image[: image.height, : image.width]
    if not np.isfinite(raw).all():
        raise RuntimeError(f"Non-finite pixels in {path}")
    return raw, path


def quality_metrics(reference, candidate) -> Fidelity:
    reference_pixels, reference_path = reference
    pixels, path = candidate
    delta = pixels.astype(np.float64) - reference_pixels
    mse = float(np.mean(delta**2))
    score = subprocess.check_output([str(METRIC), str(reference_path), str(path)], text=True)
    values = re.findall(r"-?\d+(?:\.\d+)?", score)
    return {"mse": mse, "psnr_db": -10 * math.log10(mse) if mse else None, "ssimulacra2": float(values[-1])}


def summarize(rows, batch):
    pages = {}
    for name in dict.fromkeys(row["image"] for row in rows):
        matches = [row for row in rows if row["image"] == name]
        pages[name] = {
            "real_tiles": matches[0]["real_tiles"],
            "submitted_tiles": matches[0]["submitted_tiles"],
            **{
                key: statistics.median(row[key] for row in matches)
                for key in ("page_ms", "inference_ms", "other_ms", "native_setup_ms")
            },
        }
    tiles = sum(page["real_tiles"] for page in pages.values())
    total = {
        key: sum(page[key] for page in pages.values())
        for key in ("page_ms", "inference_ms", "other_ms", "native_setup_ms")
    }
    total["real_tiles"] = tiles
    total["submitted_tiles"] = sum(page["submitted_tiles"] for page in pages.values())
    total["noncontiguous_input_calls"] = sum(not call["input_c_contiguous"] for row in rows for call in row["calls"])
    total["page_ms_per_real_tile"] = total["page_ms"] / tiles
    total["inference_ms_per_real_tile"] = total["inference_ms"] / tiles
    for stats in pages.values():
        stats["page_ms_per_real_tile"] = stats["page_ms"] / stats["real_tiles"]
        stats["inference_ms_per_real_tile"] = stats["inference_ms"] / stats["real_tiles"]
    if batch == 1:
        total["individual_tile_latency_ms"] = statistics.mean(call["run_ms"] for row in rows for call in row["calls"])
        for name, stats in pages.items():
            stats["individual_tile_latency_ms"] = statistics.mean(
                call["run_ms"] for row in rows if row["image"] == name for call in row["calls"]
            )
    return {"overall": total, "per_image": pages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manga-repo", type=Path, default=ROOT.parent / "nao-manga-rls")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "output/native-refactor/gaterv3.gtmodel")
    parser.add_argument("--onnx-dir", type=Path, default=ROOT / "output/baseline")
    parser.add_argument("--input", type=Path, default=ROOT / "samples/input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path)
    parser.add_argument("--rtx-cache", type=Path)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--overlap", type=int, nargs="+", default=[0])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--quality", action="store_true", help="Compare the batch-16 rendered pages with Torch FP32")
    args = parser.parse_args()
    if args.rounds < 1 or min(args.batch) < 1 or min(args.overlap) < 0 or max(args.overlap) >= args.tile:
        parser.error("Expected positive rounds and batches, and overlap smaller than the tile size")
    if args.quality and (16 not in args.batch or not METRIC.is_file()):
        parser.error("--quality requires batch 16 and ssimulacra2")
    args.output.mkdir(parents=True, exist_ok=True)
    args.native_cache = args.native_cache or args.output / "native-cache"
    args.rtx_cache = args.rtx_cache or args.output / "rtx-cache"
    sys.path.insert(0, str(args.manga_repo.resolve()))
    backend = importlib.import_module("nmanga.ml.onnx_backend")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if "trt-rtx" in args.variants:
        export_missing_onnx(args.onnx_dir, args.batch, args.tile)
    images = {}
    for path in sorted(args.input.iterdir()):
        if path.suffix.lower() in IMAGE_EXTS and path.is_file():
            with Image.open(path) as image:
                images[path.name] = image.copy()
    if not images:
        parser.error(f"No images found in {args.input}")
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "manga_backend": str(backend.__file__),
        "tile": args.tile,
        "rounds": args.rounds,
        "tile_timing": "numpy host input to host output, synchronous; batch 1 records actual individual tiles",
        "page_timing": "decoded PIL input to final PIL image, excluding PNG write and session creation",
        "results": {},
    }
    results = report["results"]
    for overlap in args.overlap:
        references = {}
        if args.quality:
            model, _ = load_model(CHECKPOINT, device="cuda")
            reference = TorchReference(model, torch.float32, 16, args.tile)
            references = {
                name: render_raw(backend, args, image, name, reference, 16, overlap, "torch-fp32")
                for name, image in images.items()
            }
            del reference, model
            torch.cuda.empty_cache()
        for batch in args.batch:
            variants = [v for v in args.variants if not (batch == 1 and v.endswith("-fixed"))]
            for variant in variants:
                key = f"overlap{overlap}-batch{batch}-{variant}"
                with open_session(args, variant, batch) as (session, metadata):
                    dtype = np.float16 if session.input_spec.type == "tensor(float16)" else np.float32
                    tails = set()
                    if not isinstance(session.input_spec.shape[0], int):
                        tails = {math.prod(tile_grid(image, args.tile, overlap)) % batch for image in images.values()}
                    warm_started = time.perf_counter()
                    for size in (*sorted(tails - {0}), batch, batch):
                        data = np.zeros((size, 3, args.tile, args.tile), dtype=dtype)
                        session.run([session.get_outputs()[0].name], {session.input_spec.name: data})
                    warmup = (time.perf_counter() - warm_started) * 1000
                    warmup_calls = session.rows
                    rows = [
                        measure_page(backend, image, name, session, args.tile, batch, overlap, index)
                        for index in range(args.rounds)
                        for name, image in images.items()
                    ]
                    result = {
                        "metadata": metadata,
                        "warmup_ms": warmup,
                        "warmup_calls": warmup_calls,
                        "summary": summarize(rows, batch),
                        "rows": rows,
                    }
                    if references and batch == 16:
                        fixed = batch if variant.endswith("-fixed") else None
                        quality_session = ObservedSession(session.session, fixed)
                        fidelity = {
                            name: quality_metrics(
                                references[name],
                                render_raw(
                                    backend,
                                    args,
                                    image,
                                    name,
                                    quality_session,
                                    batch,
                                    overlap,
                                    variant,
                                ),
                            )
                            for name, image in images.items()
                        }
                        sizes = {name: image.width * image.height for name, image in images.items()}
                        mse = sum(fidelity[name]["mse"] * count for name, count in sizes.items()) / sum(sizes.values())
                        result["fidelity"] = {
                            "per_image": fidelity,
                            "aggregate_psnr_db": -10 * math.log10(mse) if mse else None,
                            "mean_ssimulacra2": statistics.mean(row["ssimulacra2"] for row in fidelity.values()),
                        }
                        del quality_session
                    results[key] = result
                    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print(key, result["summary"]["overall"], flush=True)
                del session


if __name__ == "__main__":
    main()
