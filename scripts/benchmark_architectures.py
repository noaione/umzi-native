"""Benchmark the expansion checkpoints against their original Resselt CUDA FP32 forwards."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import IMAGE_EXTS, ROOT, image_tensor, tensor_image
from PIL import Image
from rtx_session import create_rtx_session
from torch import Tensor
from torch.nn import functional

from umzi_native.exporter._loader import load_checkpoint

METRIC = Path(r"C:\Tools\imaging\ssimulacra2.exe")
BACKENDS = ("fp32", "inductor", "rtx", "supplied", "eager-fp16")
RTX_WORKSPACE_BYTES = 512 * 2**20


@dataclass(frozen=True)
class Family:
    checkpoint: str
    model_class: str
    samples: str
    scale: int
    tile: int
    batch: int
    overlap: int = 0


FAMILIES = {
    "gaterv2": Family("1x_umzi_dehalftone_gater2_v1_smooth.safetensors", "GateRV2", "halftone", 1, 128, 16),
    "moesr": Family("4x_dwtp_ds_moesr_v2.pth", "MoESR", "mscale", 4, 256, 2),
    "gfisrv2": Family("4x_wtp_MangaScale_GfisrV2.pth", "GFISRV2", "mscale", 4, 256, 2, 32),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_image(path: Path) -> Tensor:
    with Image.open(path) as image:
        return tensor_image(image)


def extract(image: Tensor, top: int, left: int, tile: int) -> Tensor:
    crop = image[:, :, top : top + tile, left : left + tile]
    return functional.pad(crop, (0, tile - crop.shape[-1], 0, tile - crop.shape[-2]), mode="replicate")


def grid(height: int, width: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    step = tile - overlap
    rows, cols = (max(1, math.ceil((value - overlap) / step)) for value in (height, width))
    return [(row * step, col * step) for row in range(rows) for col in range(cols)]


def stitch(
    canvas: Tensor,
    result: Tensor,
    top: int,
    left: int,
    scale: int,
    feather: Tensor | None,
) -> None:
    top *= scale
    left *= scale
    height, width = canvas.shape[-2:]
    tile = result.shape[-1]
    if feather is not None:
        overlap = len(feather)
        if top:
            result[:, :overlap, :] *= feather[:, None]
        if top + tile < height:
            result[:, -overlap:, :] *= feather.flip(0)[:, None]
        if left:
            result[:, :, :overlap] *= feather
        if left + tile < width:
            result[:, :, -overlap:] *= feather.flip(0)
    rows, cols = min(tile, height - top), min(tile, width - left)
    canvas[:, :, top : top + rows, left : left + cols] += result[:, :rows, :cols]


def comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError(f"Output shape {candidate.shape} differs from {reference.shape}")
    squared_error = 0.0
    max_abs = 0.0
    nonfinite = 0
    for start in range(0, reference.shape[-2], 128):
        rows = (..., slice(start, start + 128), slice(None))
        block = candidate[rows]
        nonfinite += int(block.size - np.count_nonzero(np.isfinite(block)))
        delta = np.subtract(block, reference[rows], dtype=np.float64)
        squared_error += float(np.sum(np.square(delta)))
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
    if nonfinite:
        return {"finite": False, "nonfinite_values": nonfinite}
    mse = squared_error / reference.size
    return {
        "finite": True,
        "values": reference.size,
        "mse": mse,
        "psnr_db": -10 * math.log10(mse) if mse else None,
        "max_abs": max_abs,
    }


def ssimulacra2(reference: Path, candidate: Path) -> float:
    result = subprocess.run([str(METRIC), str(reference), str(candidate)], check=True, capture_output=True, text=True)
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?", result.stdout)
    if not numbers:
        raise RuntimeError(f"Cannot parse SSIMULACRA2 output: {result.stdout!r}")
    return float(numbers[-1])


def profile_placement(path: Path) -> dict[str, list[str]]:
    placement: dict[str, set[str]] = {}
    for event in json.loads(path.read_text(encoding="utf-8")):
        args = event.get("args", {})
        if event.get("cat") == "Node" and args.get("provider"):
            placement.setdefault(args["provider"], set()).add(args.get("op_name", event["name"]))
    return {provider: sorted(ops) for provider, ops in sorted(placement.items())}


class Runner:
    """Prepare device input outside the synchronized forward interval."""

    def __init__(self, run: Callable[[Tensor], Tensor], dtype: torch.dtype):
        self.run = run
        self.dtype = dtype

    def prepare(self, host: Tensor) -> Tensor:
        return host.to(device="cuda", dtype=self.dtype)

    def measure(self, host: Tensor) -> tuple[Tensor, float]:
        sample = self.prepare(host)
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = self.run(sample)
        torch.cuda.synchronize()
        return prediction, time.perf_counter() - started


def stress_inputs(tile: int, paths: list[Path], batch: int) -> Iterator[tuple[str, Tensor]]:
    shape = (1, 3, tile, tile)
    for label, gray in (("black", 0.0), ("gray", 0.5), ("gray-128", 128 / 255), ("white", 1.0)):
        yield label, torch.full(shape, gray)
    index = torch.arange(tile)
    checker = (index[:, None] + index[None, :]).remainder(2)
    yield "checkerboard", checker.float().expand(shape).contiguous()
    impulse = torch.zeros(shape)
    impulse[0, 0, 0, 1] = 1
    impulse[0, 1, -2, -1] = 0.75
    impulse[0, 2, 1, -2] = 0.5
    yield "border-impulses", impulse
    generator = torch.Generator().manual_seed(20261122)
    yield "noise", torch.rand(shape, generator=generator)
    for path in paths:
        image = read_image(path)
        height, width = image.shape[-2:]
        yield f"sample-{path.stem}", extract(image, max(0, (height - tile) // 2), max(0, (width - tile) // 2), tile)
    for height, width in ((65, 79), (128, 160)):
        yield f"rect-{height}x{width}", torch.rand((1, 3, height, width), generator=generator)
    if batch > 1:
        yield "distinct-batch", torch.rand((batch, 3, tile, tile), generator=generator)


def validate_stress(
    runners: dict[int, Runner], paths: list[Path], family: Family, batch: int, backend: str, base: Path
):
    reference = backend == "fp32"
    folder = base / "fp32" / "stress"
    if reference:
        folder.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {}
    for label, sample in stress_inputs(family.tile, paths, batch):
        count, _, height, width = sample.shape
        row: dict[str, Any] = {"input_shape": list(sample.shape)}
        rows[label] = row
        if count not in runners or (backend == "rtx" and (height, width) != (family.tile, family.tile)):
            row["skipped"] = "outside the fixed RTX input shapes"
            continue
        try:
            pred, seconds = runners[count].measure(sample)
            expected = (count, 3, height * family.scale, width * family.scale)
            if tuple(pred.shape) != expected:
                raise ValueError(f"Expected stress output {expected}, got {tuple(pred.shape)}")
            output = pred.float().cpu().numpy()
            row.update(output_shape=list(expected), first_call_seconds=seconds)
            path = folder / f"{label}.npy"
            if reference:
                if not np.isfinite(output).all():
                    raise ValueError("Non-finite FP32 reference")
                np.save(path, output)
                row["finite"] = True
            else:
                row.update(comparison(output, np.load(path, mmap_mode="r")))
            if count > 1:
                row["per_sample_vs_individual"] = [
                    comparison(item, runners[1].run(runners[1].prepare(part))[0].float().cpu().numpy())
                    for item, part in zip(output, sample.split(1), strict=True)
                ]
        except Exception as error:
            row.update(error_type=type(error).__name__, error=str(error))
            if reference:
                raise
        outcome = row.get("error_type") or (
            "non-finite" if row.get("finite") is False else row.get("psnr_db", "finite")
        )
        print(f"  {label}: {outcome}", flush=True)
    return rows


def check_tiny(runner: Runner, scale: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for height, width in ((1, 1), (2, 2), (3, 5), (8, 9)):
        try:
            output, _ = runner.measure(torch.zeros((1, 3, height, width)))
        except RuntimeError as error:
            if "Padding size should be less" not in str(error):
                raise
            rows[f"{height}x{width}"] = {"padding_error": str(error)}
            continue
        if tuple(output.shape) != (1, 3, height * scale, width * scale) or not torch.isfinite(output).all():
            raise ValueError(f"Bad tiny output for {height}x{width}")
        rows[f"{height}x{width}"] = {"output_shape": list(output.shape), "finite": True}
    return rows


def timed_page(
    image: Tensor, label: str, runners: dict[int, Runner], family: Family, batch: int
) -> tuple[dict[str, Any], Tensor]:
    height, width = image.shape[-2:]
    tile, overlap, scale = family.tile, family.overlap, family.scale
    coords = grid(height, width, tile, overlap)
    output = torch.zeros((1, 3, height * scale, width * scale))
    feather = (torch.arange(overlap * scale) + 0.5) / (overlap * scale) if overlap else None
    calls = []
    for start in range(0, len(coords), batch):
        group = coords[start : start + batch]
        inputs = [extract(image, top, left, tile) for top, left in group]
        # Repeat the last real input so every backend submits the same fixed batch.
        sample = torch.cat(inputs + [inputs[-1]] * (batch - len(group)))
        prediction, elapsed = runners[batch].measure(sample)
        if prediction.shape != (batch, 3, tile * scale, tile * scale):
            raise ValueError(f"Incorrect page tile output shape {list(prediction.shape)}")
        calls.append(elapsed)
        prediction = prediction.to(device="cpu", dtype=torch.float32)
        for part, (top, left) in zip(prediction[: len(group)], group, strict=True):
            stitch(output, part, top, left, scale, feather)
    latencies: list[dict[str, Any]] = []
    for index, (top, left) in enumerate(coords):
        _, elapsed = runners[1].measure(extract(image, top, left, tile))
        latencies.append({
            "image": label,
            "tile_index": index,
            "top": top,
            "left": left,
            "milliseconds": elapsed * 1000,
        })
    seconds = sum(calls)
    return {
        "image": label,
        "height": height,
        "width": width,
        "input_pixels": height * width,
        "real_tiles": len(coords),
        "submitted_tiles": len(calls) * batch,
        "batch_inference_seconds": seconds,
        "batch_input_mpx_per_second": height * width / seconds / 1e6,
        "batch_call_seconds": calls,
        "average_batch1_tile_latency_ms": statistics.mean(sample["milliseconds"] for sample in latencies),
        "batch1_tile_latency": latencies,
    }, output


def torch_runners(checkpoint: Path, family: Family, batch: int, backend: str, report: dict[str, Any]):
    model, _ = load_checkpoint(checkpoint)
    if type(model).__name__ != family.model_class:
        raise ValueError(f"Unexpected checkpoint class {type(model)}")
    report["model_class"] = type(model).__name__
    report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    dtype = torch.float32 if backend == "fp32" else torch.float16
    model.eval().to(device="cuda", dtype=dtype)
    run = model
    if backend == "inductor":
        if os.name == "nt":
            # Windows cannot pass file descriptors to Inductor's compile subprocesses.
            os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        torch._dynamo.reset()
        torch._dynamo.config.suppress_errors = False
        report["compile"] = {
            "backend": "inductor",
            "dynamic": False,
            "fullgraph": True,
            "mode": "default",
            "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
            "numerics": os.environ.get("TORCHINDUCTOR_NUMERICS", "default"),
        }
        run = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
    runner = Runner(run, dtype)
    runners = dict.fromkeys({1, batch}, runner)
    for count in sorted({1, batch}):
        print(f"Preparing {backend} batch {count}", flush=True)
        sample = torch.zeros((count, 3, family.tile, family.tile))
        stage = report["setup"].setdefault(str(count), {"input_shape": list(sample.shape)})
        stage["first_call_seconds"] = runner.measure(sample)[1]
        stage["warm_call_seconds"] = runner.measure(sample)[1]
        stage["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return runners


class RtxRunner(Runner):
    def __init__(self, session: Any, batch: int, tile: int, scale: int):
        self.session = session
        self.input = torch.empty((batch, 3, tile, tile), dtype=torch.float16, device="cuda")
        self.output = torch.empty((batch, 3, tile * scale, tile * scale), dtype=torch.float16, device="cuda")
        self.binding = session.io_binding()
        self.binding.bind_input(
            session.get_inputs()[0].name, "cuda", 0, np.float16, tuple(self.input.shape), self.input.data_ptr()
        )
        self.binding.bind_output(
            session.get_outputs()[0].name, "cuda", 0, np.float16, tuple(self.output.shape), self.output.data_ptr()
        )
        super().__init__(self.forward, torch.float16)

    def prepare(self, host: Tensor) -> Tensor:
        return self.input.copy_(host.to(dtype=self.dtype))

    def forward(self, sample: Tensor) -> Tensor:
        self.session.run_with_iobinding(self.binding)
        return self.output


def prepare_rtx_graph(checkpoint: Path, directory: Path, batch: int, tile: int, scale: int):
    import onnx

    path = directory / f"t{tile}-b{batch}.onnx"
    model, _ = load_checkpoint(checkpoint)
    model.eval().half().cuda()
    sample = torch.zeros((batch, 3, tile, tile), dtype=torch.float16, device="cuda")
    print(f"Exporting {path.name}", flush=True)
    torch.onnx.export(
        model, (sample,), str(path), dynamo=False, opset_version=20, input_names=["input"], output_names=["output"]
    )
    graph = onnx.load(path, load_external_data=False)
    inp, out = graph.graph.input[0], graph.graph.output[0]
    for value in (inp, out):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
            raise ValueError(f"Expected FP16 I/O in {path}")
    # The fixed-shape legacy exporter leaves the final Slice's axes symbolic.
    for axis, extent in zip(out.type.tensor_type.shape.dim, (batch, 3, tile * scale, tile * scale), strict=True):
        if axis.HasField("dim_value") and axis.dim_value != extent:
            raise ValueError(f"Unexpected exported output axis {axis} in {path}")
        axis.dim_value = extent
    # The legacy exporter can copy PyTorch's negative permute axes into ONNX Transpose.
    normalized_permutations = 0
    for node in graph.graph.node:
        if node.op_type != "Transpose":
            continue
        for attribute in node.attribute:
            if attribute.name != "perm" or all(axis >= 0 for axis in attribute.ints):
                continue
            rank = len(attribute.ints)
            axes = [axis + rank if axis < 0 else axis for axis in attribute.ints]
            if sorted(axes) != list(range(rank)):
                raise ValueError(f"Invalid transpose permutation in {node.name}: {list(attribute.ints)}")
            attribute.ints[:] = axes
            normalized_permutations += 1
    onnx.checker.check_model(graph, full_check=True)
    onnx.save_model(graph, path)
    return path, inp.name, normalized_permutations


def open_rtx_runner(path: Path, input_name: str, batch: int, family: Family, stage: dict[str, Any]) -> Runner:
    tile, scale = family.tile, family.scale
    cache = path.parent / f"cache-{batch}"
    started = time.perf_counter()
    print(f"Profiling {path.name}", flush=True)
    session, _ = create_rtx_session(
        path,
        batch,
        tile,
        cache,
        path.parent / f"placement-b{batch}",
        scale=scale,
        input_name=input_name,
        cuda_graph=False,
        workspace_bytes=RTX_WORKSPACE_BYTES,
    )
    shape = (batch, 3, tile, tile)
    try:
        output = np.asarray(session.run(None, {input_name: np.zeros(shape, dtype=np.float16)})[0])
        if output.shape != (batch, 3, tile * scale, tile * scale) or output.dtype != np.float16:
            raise ValueError(f"Unexpected RTX output {output.shape} with dtype {output.dtype}")
    finally:
        profile = Path(session.end_profiling())
        stage.update(
            placement_profile=profile.name,
            providers=session.get_providers(),
            node_placement=profile_placement(profile),
            profile_seconds=time.perf_counter() - started,
        )
    del session
    if list(stage["node_placement"]) != ["NvTensorRTRTXExecutionProvider"]:
        raise RuntimeError(f"Unexpected TensorRT-RTX node placement: {stage['node_placement']}")
    session, config = create_rtx_session(
        path, batch, tile, cache, scale=scale, input_name=input_name, workspace_bytes=RTX_WORKSPACE_BYTES
    )
    stage["provider_options"] = config
    runner = RtxRunner(session, batch, tile, scale)
    sample = torch.zeros(shape)
    stage["first_call_seconds"] = runner.measure(sample)[1]
    stage["warm_call_seconds"] = runner.measure(sample)[1]
    stage["device_free_bytes_after_warmup"] = torch.cuda.mem_get_info()[0]
    return runner


def rtx_runners(checkpoint: Path, directory: Path, family: Family, batch: int, report: dict[str, Any]):
    runners: dict[int, Runner] = {}
    errors = []
    for count in sorted({1, batch}):
        tile = family.tile
        stage = report["setup"].setdefault(str(count), {"input_shape": [count, 3, tile, tile]})
        try:
            started = time.perf_counter()
            path, input_name, permutations = prepare_rtx_graph(checkpoint, directory, count, tile, family.scale)
            stage.update(
                export_seconds=time.perf_counter() - started,
                graph=path.name,
                graph_sha256=digest(path),
                normalized_transpose_permutations=permutations,
            )
            torch.cuda.empty_cache()
            runners[count] = open_rtx_runner(path, input_name, count, family, stage)
        except Exception as error:
            stage.update(error_type=type(error).__name__, error=str(error))
            errors.append(f"batch {count}: {type(error).__name__}")
            traceback.print_exc()
    if errors:
        raise RuntimeError(f"RTX setup failed for {', '.join(errors)}. See each shape's setup record.")
    return runners


def supplied_graph(directory: Path, family: Family, paths: list[Path], batch: int, base: Path, report: dict[str, Any]):
    import onnx
    import onnxruntime as ort

    path = ROOT / "models/4x_dwtp_ds_moesr_v2_op20_fp16.onnx"
    graph = onnx.load(path, load_external_data=False)
    report.update(
        graph_sha256=digest(path),
        graph_metadata={item.key: item.value for item in graph.metadata_props},
        opset=[item.version for item in graph.opset_import],
    )
    ort.disable_telemetry_events()
    ort.preload_dlls()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = str(directory / "placement")
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[
            (
                "CUDAExecutionProvider",
                {"use_tf32": "0", "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0"},
            ),
            "CPUExecutionProvider",
        ],
        enable_fallback=False,
    )
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    report.update(
        graph_input={"name": inp.name, "type": inp.type, "shape": inp.shape},
        graph_output={"name": out.name, "type": out.type, "shape": out.shape},
        providers=session.get_providers(),
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDAExecutionProvider did not activate for supplied graph")
    rows: dict[str, Any] = {}
    report["stress"] = rows
    try:
        for label, sample in stress_inputs(family.tile, paths, batch):
            ref = base / "fp32" / "stress" / f"{label}.npy"
            row: dict[str, Any] = {"input_shape": list(sample.shape)}
            rows[label] = row
            try:
                output = np.asarray(session.run([out.name], {inp.name: sample.numpy()})[0])
                row.update(output_shape=list(output.shape), **comparison(output, np.load(ref, mmap_mode="r")))
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error))
            print(f"  {label}: {rows[label]}", flush=True)
    finally:
        profile = Path(session.end_profiling())
        report.update(placement_profile=profile.name, node_placement=profile_placement(profile))
    if "Conv" not in report["node_placement"].get("CUDAExecutionProvider", ()):
        raise RuntimeError("The supplied graph did not execute convolution on the CUDA provider")


def summarize(pages: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = sum(page["batch_inference_seconds"] for page in pages)
    latencies = [item["milliseconds"] for page in pages for item in page["batch1_tile_latency"]]
    summary = {
        "batch_inference_seconds": elapsed,
        "batch_input_mpx_per_second": sum(page["input_pixels"] for page in pages) / elapsed / 1e6,
        "real_tiles": len(latencies),
        "submitted_tiles": sum(page["submitted_tiles"] for page in pages),
        "average_batch1_tile_latency_ms": statistics.mean(latencies),
    }
    if all("ssimulacra2" in page for page in pages):
        values = sum(page["values"] for page in pages)
        mse = sum(page["mse"] * page["values"] for page in pages) / values
        summary.update(
            mse=mse,
            psnr_db=-10 * math.log10(mse) if mse else None,
            max_abs=max(page["max_abs"] for page in pages),
            ssimulacra2_mean=statistics.mean(page["ssimulacra2"] for page in pages),
            ssimulacra2_min=min(page["ssimulacra2"] for page in pages),
        )
    return summary


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("backend", choices=BACKENDS)
    parser.add_argument("--batch", type=int, help="Override the family's planned throughput batch")
    parser.add_argument("--input", type=Path, help="Override the family's default sample directory")
    parser.add_argument("--output", type=Path, default=ROOT / "output/architecture-baselines")
    args = parser.parse_args()
    family = FAMILIES[args.family]
    batch = family.batch if args.batch is None else args.batch
    if batch < 1:
        parser.error("--batch must be positive")
    if args.backend == "rtx" and args.family == "gfisrv2":
        parser.error("The baseline matrix skips TensorRT-RTX for GFISRv2")
    if args.backend == "supplied" and args.family != "moesr":
        parser.error("Only MoESR has a supplied graph")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    inputs = args.input or ROOT / "samples" / family.samples
    paths = sorted(p for p in inputs.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths or len({path.stem for path in paths}) != len(paths):
        parser.error("--input must contain images with unique filename stems")
    checkpoint = ROOT / "models" / family.checkpoint
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required to record the CUDA driver and device memory")
    base = args.output.resolve() / args.family / f"batch-{batch}"
    directory = base / args.backend
    directory.mkdir(parents=True, exist_ok=True)
    signature = {
        "checkpoint": family.checkpoint,
        "checkpoint_sha256": digest(checkpoint),
        "batch": batch,
        "tile": family.tile,
        "overlap": family.overlap,
        "scale": family.scale,
        "images": {path.name: digest(path) for path in paths},
        "tiling": "row-major, replicate edge padding, linear output-space feather, crop to source size times scale",
    }
    report: dict[str, Any] = {
        "config": signature,
        "backend": args.backend,
        "status": "running",
        "setup": {},
        "pages": [],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "resselt": version("resselt"),
            "triton": version("triton-windows"),
            "onnx": version("onnx"),
            "ort": version("onnxruntime-gpu"),
            "ort_rtx_plugin": version("onnxruntime-ep-nv-tensorrt-rtx-cu13"),
            "device": torch.cuda.get_device_name(0),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_smi": subprocess.check_output(
                [nvidia_smi, "--query-gpu=driver_version,memory.used,memory.total", "--format=csv"], text=True
            ).strip(),
            "tf32": False,
            "cudnn_benchmark": False,
            "cpu_threads": torch.get_num_threads(),
        },
        "timing": "synchronized wall time of device-resident forward calls; excludes transfers, tiling and rendering",
        "vram": {"free_before_bytes": torch.cuda.mem_get_info()[0]},
    }
    reference = args.backend == "fp32"

    def save() -> None:
        path = directory / "report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    save()
    try:
        if not reference:
            previous = json.loads((base / "fp32" / "report.json").read_text(encoding="utf-8"))
            if previous["status"] != "complete" or previous["config"] != signature:
                raise ValueError("The completed FP32 report must match this run's checkpoint, inputs and tiling")
        if args.backend == "supplied":
            supplied_graph(directory, family, paths, batch, base, report)
            if any(row.get("finite") is not True for row in report["stress"].values()):
                raise RuntimeError("Supplied-graph validation failed. See individual stress cases.")
            report["status"] = "complete"
            return
        if args.backend == "rtx":
            runners = rtx_runners(checkpoint, directory, family, batch, report)
        else:
            runners = torch_runners(checkpoint, family, batch, args.backend, report)
        if reference:
            report["tiny_inputs"] = check_tiny(runners[1], family.scale)
        report["stress"] = validate_stress(runners, paths, family, batch, args.backend, base)
        stress_comparisons = {label: row for label, row in report["stress"].items() if row.get("mse") is not None}
        if stress_comparisons:
            report["worst_stress_psnr"] = max(stress_comparisons, key=lambda label: stress_comparisons[label]["mse"])
        save()
        for path in paths:
            row, output = timed_page(read_image(path), path.name, runners, family, batch)
            png = directory / f"{path.stem}.png"
            if reference:
                if not torch.isfinite(output).all():
                    raise ValueError(f"Non-finite FP32 page {path.name}")
                np.save(directory / f"{path.stem}.npy", output.numpy())
            else:
                raw = np.load(base / "fp32" / f"{path.stem}.npy", mmap_mode="r")
                row.update(comparison(output.numpy(), raw))
            if reference or row["finite"]:
                image_tensor(output).save(png)
            if not reference and row["finite"]:
                row["ssimulacra2"] = ssimulacra2(base / "fp32" / png.name, png)
            report["pages"].append(row)
            print(
                f"  {path.name}: {row['batch_inference_seconds']:.3f}s, "
                f"{row['average_batch1_tile_latency_ms']:.3f}ms/tile",
                flush=True,
            )
            save()
        report["summary"] = summarize(report["pages"])
        stress_failed = any(row.get("error") or row.get("finite") is False for row in report["stress"].values())
        pages_failed = any(row.get("finite") is False for row in report["pages"])
        if pages_failed:
            report["status"] = "page_fidelity_failed"
        elif stress_failed:
            report["status"] = "stress_failed"
        else:
            report["status"] = "complete"
    except Exception as error:
        report.update(status="blocked", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        report["vram"].update(
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            free_after_bytes=torch.cuda.mem_get_info()[0],
        )
        save()
        print(f"{report['status']}: {directory / 'report.json'}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
