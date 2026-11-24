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
from contextlib import ExitStack
from dataclasses import dataclass, replace
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import IMAGE_EXTS, ROOT, image_tensor, tensor_image
from PIL import Image
from reference.gaterv2 import use_fp32_attention
from reference.native_session import DeviceSession
from reference.nvml import Nvml, NvmlSampler
from rtx_session import create_rtx_session
from torch import Tensor
from torch.nn import functional

from umzi_native import InferenceSession, SessionOptions, peek_model
from umzi_native.exporter._loader import load_checkpoint

METRIC = Path(r"C:\Tools\imaging\ssimulacra2.exe")
NATIVE_BACKENDS = ("native-mixed", "native-fp16", "native-fp32")
RTX_BACKENDS = ("rtx", "rtx-fp32")
BACKENDS = ("fp32", "inductor", "inductor-attention-fp32", "supplied", "eager-fp16", *RTX_BACKENDS, *NATIVE_BACKENDS)
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
    "smosr": Family("2x_enhancr_da_smosr_v1.safetensors", "SMoSR", "2x-test", 2, 256, 1),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_image(path: Path) -> Tensor:
    with Image.open(path) as image:
        return tensor_image(image)


def extract(image: Tensor, top: int, left: int, tile: int, *, variable_edges: bool = False) -> Tensor:
    crop = image[:, :, top : top + tile, left : left + tile]
    minimum = 3 if variable_edges else tile
    height, width = crop.shape[-2:]
    return functional.pad(crop, (0, max(0, minimum - width), 0, max(0, minimum - height)), mode="replicate")


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
    tile_h, tile_w = result.shape[-2:]
    if feather is not None:
        overlap = len(feather)
        if top:
            result[:, :overlap, :] *= feather[:, None]
        if top + tile_h < height:
            result[:, -overlap:, :] *= feather.flip(0)[:, None]
        if left:
            result[:, :, :overlap] *= feather
        if left + tile_w < width:
            result[:, :, -overlap:] *= feather.flip(0)
    rows, cols = min(tile_h, height - top), min(tile_w, width - left)
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


def score_stress_render(directory: Path, label: str, reference: Tensor, candidate: Tensor) -> float:
    directory.mkdir(parents=True, exist_ok=True)
    reference_png = directory / f"{label}-fp32.png"
    candidate_png = directory / f"{label}-candidate.png"
    image_tensor(reference).save(reference_png)
    image_tensor(candidate).save(candidate_png)
    return ssimulacra2(reference_png, candidate_png)


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


class NativeRunner(Runner):
    """Reuse device input and output storage for the synchronized benchmark."""

    def __init__(self, session: InferenceSession):
        runner = DeviceSession(session, reuse_output=True)
        super().__init__(runner, runner.dtype)
        self.input: Tensor | None = None

    def prepare(self, host: Tensor) -> Tensor:
        if self.input is None or self.input.shape != host.shape:
            self.input = torch.empty(host.shape, device="cuda", dtype=self.dtype)
        return self.input.copy_(host.to(dtype=self.dtype))


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
        if count not in runners or (backend in RTX_BACKENDS and (height, width) != (family.tile, family.tile)):
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
                expected = np.load(path, mmap_mode="r")
                row.update(comparison(output, expected))
                if backend == "inductor-attention-fp32" and label == "checkerboard" and row["finite"]:
                    row["ssimulacra2"] = score_stress_render(
                        base / backend / "stress-renders",
                        label,
                        torch.from_numpy(expected.copy()),
                        torch.from_numpy(output),
                    )
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
    image: Tensor,
    label: str,
    runners: dict[int, Runner],
    family: Family,
    batch: int,
    *,
    shape_runners: dict[tuple[int, int], Runner] | None = None,
) -> tuple[dict[str, Any], Tensor]:
    variable_edges = shape_runners is not None
    if variable_edges and (batch != 1 or family.overlap):
        raise ValueError("Variable edge tiles require batch 1 and zero overlap")
    height, width = image.shape[-2:]
    tile, overlap, scale = family.tile, family.overlap, family.scale
    coords = grid(height, width, tile, overlap)
    output = torch.zeros((1, 3, height * scale, width * scale))
    feather = (torch.arange(overlap * scale) + 0.5) / (overlap * scale) if overlap else None
    calls = []
    input_sizes = []
    for start in range(0, len(coords), batch):
        group = coords[start : start + batch]
        inputs = [extract(image, top, left, tile, variable_edges=variable_edges) for top, left in group]
        # Repeat the last real input so every backend submits the same fixed batch.
        sample = torch.cat(inputs + [inputs[-1]] * (batch - len(group)))
        input_h, input_w = sample.shape[-2:]
        runner = runners[batch] if shape_runners is None else shape_runners[input_h, input_w]
        prediction, elapsed = runner.measure(sample)
        if prediction.shape != (batch, 3, input_h * scale, input_w * scale):
            raise ValueError(f"Incorrect page tile output shape {list(prediction.shape)}")
        calls.append(elapsed)
        input_sizes.append((input_h, input_w))
        prediction = prediction.to(device="cpu", dtype=torch.float32)
        for part, (top, left) in zip(prediction[: len(group)], group, strict=True):
            stitch(output, part, top, left, scale, feather)
    latencies: list[dict[str, Any]] = []
    for index, (top, left) in enumerate(coords):
        if batch == 1:
            elapsed = calls[index]
        else:
            _, elapsed = runners[1].measure(extract(image, top, left, tile))
        latencies.append({
            "image": label,
            "tile_index": index,
            "top": top,
            "left": left,
            "milliseconds": elapsed * 1000,
            "input_hw": input_sizes[index] if batch == 1 else (tile, tile),
        })
    seconds = sum(calls)
    tile_ms = [sample["milliseconds"] for sample in latencies]
    return {
        "image": label,
        "height": height,
        "width": width,
        "input_pixels": height * width,
        "submitted_input_pixels": sum(h * w * batch for h, w in input_sizes),
        "real_tiles": len(coords),
        "submitted_tiles": len(calls) * batch,
        "batch_inference_seconds": seconds,
        "image_inference_ms": seconds * 1000,
        "batch_input_mpx_per_second": height * width / seconds / 1e6,
        "batch_call_seconds": calls,
        "average_batch1_tile_latency_ms": statistics.mean(tile_ms),
        "median_batch1_tile_latency_ms": statistics.median(tile_ms),
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
    if backend == "inductor-attention-fp32":
        use_fp32_attention(model)
        report["precision"] = "FP16 except FP32 latent attention normalization, contractions and division"
    if backend in ("inductor", "inductor-attention-fp32"):
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
    def __init__(self, session: Any, batch: int, tile: int, scale: int, dtype: torch.dtype):
        self.session = session
        self.input = torch.empty((batch, 3, tile, tile), dtype=dtype, device="cuda")
        self.output = torch.empty((batch, 3, tile * scale, tile * scale), dtype=dtype, device="cuda")
        element_type = np.float32 if dtype == torch.float32 else np.float16
        self.binding = session.io_binding()
        self.binding.bind_input(
            session.get_inputs()[0].name, "cuda", 0, element_type, tuple(self.input.shape), self.input.data_ptr()
        )
        self.binding.bind_output(
            session.get_outputs()[0].name, "cuda", 0, element_type, tuple(self.output.shape), self.output.data_ptr()
        )
        super().__init__(self.forward, dtype)

    def prepare(self, host: Tensor) -> Tensor:
        return self.input.copy_(host.to(dtype=self.dtype))

    def forward(self, sample: Tensor) -> Tensor:
        self.session.run_with_iobinding(self.binding)
        return self.output


def prepare_rtx_graph(checkpoint: Path, directory: Path, batch: int, tile: int, scale: int, dtype: torch.dtype):
    import onnx

    path = directory / f"t{tile}-b{batch}.onnx"
    model, _ = load_checkpoint(checkpoint)
    model.eval().to(device="cuda", dtype=dtype)
    sample = torch.zeros((batch, 3, tile, tile), dtype=dtype, device="cuda")
    print(f"Exporting {path.name}", flush=True)
    torch.onnx.export(
        model, (sample,), str(path), dynamo=False, opset_version=20, input_names=["input"], output_names=["output"]
    )
    graph = onnx.load(path, load_external_data=False)
    inp, out = graph.graph.input[0], graph.graph.output[0]
    element_type = onnx.TensorProto.FLOAT if dtype == torch.float32 else onnx.TensorProto.FLOAT16
    for value in (inp, out):
        if value.type.tensor_type.elem_type != element_type:
            raise ValueError(f"Expected {dtype} I/O in {path}")
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


def open_rtx_runner(
    path: Path,
    input_name: str,
    batch: int,
    family: Family,
    stage: dict[str, Any],
    *,
    output_dir: Path | None = None,
    dtype: torch.dtype = torch.float16,
) -> Runner:
    tile, scale = family.tile, family.scale
    directory = path.parent if output_dir is None else output_dir
    cache = directory / f"cache-{batch}"
    started = time.perf_counter()
    print(f"Profiling {path.name}", flush=True)
    session, _ = create_rtx_session(
        path,
        batch,
        tile,
        cache,
        directory / f"placement-b{batch}",
        scale=scale,
        input_name=input_name,
        cuda_graph=False,
        workspace_bytes=RTX_WORKSPACE_BYTES,
    )
    shape = (batch, 3, tile, tile)
    element_type = np.float32 if dtype == torch.float32 else np.float16
    try:
        output = np.asarray(session.run(None, {input_name: np.zeros(shape, dtype=element_type)})[0])
        if output.shape != (batch, 3, tile * scale, tile * scale) or output.dtype != element_type:
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
    runner = RtxRunner(session, batch, tile, scale, dtype)
    sample = torch.zeros(shape)
    stage["first_call_seconds"] = runner.measure(sample)[1]
    stage["warm_call_seconds"] = runner.measure(sample)[1]
    stage["device_free_bytes_after_warmup"] = torch.cuda.mem_get_info()[0]
    return runner


def rtx_runners(checkpoint: Path, directory: Path, family: Family, batch: int, report: dict[str, Any]):
    runners: dict[int, Runner] = {}
    errors = []
    dtype = torch.float32 if report["backend"] == "rtx-fp32" else torch.float16
    for count in sorted({1, batch}):
        tile = family.tile
        stage = report["setup"].setdefault(str(count), {"input_shape": [count, 3, tile, tile]})
        try:
            started = time.perf_counter()
            path, input_name, permutations = prepare_rtx_graph(checkpoint, directory, count, tile, family.scale, dtype)
            stage.update(
                export_seconds=time.perf_counter() - started,
                graph=path.name,
                graph_sha256=digest(path),
                normalized_transpose_permutations=permutations,
            )
            torch.cuda.empty_cache()
            runners[count] = open_rtx_runner(path, input_name, count, family, stage, dtype=dtype)
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
        "median_batch1_tile_latency_ms": statistics.median(latencies),
        "median_image_inference_ms": statistics.median(page["batch_inference_seconds"] * 1000 for page in pages),
        "median_image_average_batch1_tile_latency_ms": statistics.median(
            page["average_batch1_tile_latency_ms"] for page in pages
        ),
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
    if all("vram" in page for page in pages):
        summary["peak_page_device_used_mib"] = max(page["vram"]["device_peak_mib"] for page in pages)
        summary["peak_page_above_background_mib"] = max(
            page["vram"]["peak_above_setup_background_mib"] for page in pages
        )
    return summary


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("backend", choices=BACKENDS)
    parser.add_argument("--batch", type=int, help="Override the family's planned throughput batch")
    parser.add_argument("--tile", type=int, help="Override the square input tile size in pixels")
    parser.add_argument("--variable-edge-tiles", action="store_true", help="Submit rectangular SMoSR edge crops")
    parser.add_argument("--overlap", type=int, help="Override the family's default tile overlap in input pixels")
    parser.add_argument("--input", type=Path, help="Override the family's default sample directory")
    parser.add_argument("--output", type=Path, default=ROOT / "output/architecture-baselines")
    parser.add_argument("--native-model", type=Path, help="Exported SMoSR .ummodel")
    parser.add_argument("--native-library", type=Path, help="Umzi session library for the native backend")
    parser.add_argument("--native-cache-dir", type=Path, help="SMoSR native plan cache directory")
    parser.add_argument("--tag", "--native-tag", help="Suffix for a named benchmark run")
    parser.add_argument("--tune-convolutions", action="store_true", help="Time the dominant zero-workspace SMoSR plans")
    parser.add_argument(
        "--mixed-fp16-accumulation", action="store_true", help="Use FP16 accumulation in SMoSR mixed body convolutions"
    )
    parser.add_argument("--fp16-fuse-silu", action="store_true", help="Fuse SMoSR FP16 convolutions with bias and SiLU")
    args = parser.parse_args()
    family = FAMILIES[args.family]
    if args.tile is not None:
        if args.tile < 1:
            parser.error("--tile must be positive")
        family = replace(family, tile=args.tile)
    if args.overlap is not None:
        family = replace(family, overlap=args.overlap)
    if not 0 <= family.overlap < family.tile:
        parser.error(f"Overlap must satisfy 0 <= overlap < {family.tile}")
    batch = family.batch if args.batch is None else args.batch
    if batch < 1:
        parser.error("--batch must be positive")
    if args.backend in RTX_BACKENDS and args.family == "gfisrv2":
        parser.error("The baseline matrix skips TensorRT-RTX for GFISRv2")
    if args.backend == "supplied" and args.family != "moesr":
        parser.error("Only MoESR has a supplied graph")
    if args.backend == "inductor-attention-fp32" and args.family != "gaterv2":
        parser.error("The FP32 attention candidate requires GateRV2")
    native = args.backend in NATIVE_BACKENDS
    if args.variable_edge_tiles and (
        args.family != "smosr"
        or batch != 1
        or family.overlap
        or family.tile < 3
        or (not native and args.backend != "fp32")
    ):
        parser.error("Variable edge tiles require SMoSR native or Torch FP32, batch 1, tile >= 3, and zero overlap")
    if args.native_cache_dir and not native:
        parser.error("--native-cache-dir requires a native backend")
    if native and (args.family != "smosr" or batch != 1 or not args.native_model or not args.native_library):
        parser.error("Native baselines require SMoSR batch 1, --native-model, and --native-library")
    if args.tag is not None:
        if not native and args.backend not in RTX_BACKENDS:
            parser.error("--tag requires a native or TensorRT-RTX backend")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag):
            parser.error("--tag requires a name using letters, digits, underscores, or hyphens")
    if args.tune_convolutions and (not native or args.backend == "native-fp32"):
        parser.error("--tune-convolutions requires a native mixed or FP16 backend")
    if args.mixed_fp16_accumulation and args.backend != "native-mixed":
        parser.error("--mixed-fp16-accumulation requires the native mixed backend")
    if args.fp16_fuse_silu and args.backend != "native-fp16":
        parser.error("--fp16-fuse-silu requires the native FP16 backend")
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
    if family.tile != FAMILIES[args.family].tile:
        base /= f"tile-{family.tile}"
    if args.variable_edge_tiles:
        base /= "variable-edges"
    directory = base / (f"{args.backend}-{args.tag}" if args.tag else args.backend)
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
    if args.variable_edge_tiles:
        signature["tiling"] = "row-major, exact edge crops, replicate to minimum 3, crop to source size times scale"
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
        "timing": "synchronized device-resident forward wall time, excluding host transfers, tiling and rendering",
        "vram": {"free_before_bytes": torch.cuda.mem_get_info()[0]},
    }
    reference = args.backend == "fp32"
    stack = ExitStack()
    session = None
    options = None

    def save() -> None:
        path = directory / "report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    save()
    runners: dict[int, Runner]
    try:
        # Match the CUDA device even when CUDA_VISIBLE_DEVICES remaps its index.
        device_uuid = str(torch.cuda.get_device_properties(0).uuid)
        nvml = Nvml(device_uuid if device_uuid.startswith("GPU-") else f"GPU-{device_uuid}")
        stack.callback(nvml.close)
        background = nvml.read()
        report["vram"]["before_backend_setup"] = background
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
        if args.backend in RTX_BACKENDS:
            runners = rtx_runners(checkpoint, directory, family, batch, report)
        elif native:
            packaged = peek_model(args.native_model)
            if (
                not packaged.supported
                or packaged.family_name != args.family
                or getattr(packaged.payload, "checkpoint", None) != signature["checkpoint_sha256"]
            ):
                raise ValueError("The native package must match the benchmark checkpoint")
            _, precision = args.backend.split("-", 1)
            kernel_options = {"tune_convolutions": 1} if args.tune_convolutions else {}
            if args.mixed_fp16_accumulation:
                kernel_options["fp16_accumulation"] = 1
            if args.fp16_fuse_silu:
                kernel_options["fuse_silu"] = 1
            report["native"] = {
                "precision": precision,
                "package_sha256": packaged.digest,
                "library_sha256": digest(args.native_library),
                "kernel_options": kernel_options,
            }
            if args.native_cache_dir:
                report["native"]["cache_dir"] = str(args.native_cache_dir.resolve())
            options = SessionOptions(
                precision=precision,
                dll_path=args.native_library,
                kernel_options=kernel_options,
                cache_dir=args.native_cache_dir,
            )
            session = stack.enter_context(InferenceSession(args.native_model, options))
            session.prepare((1, 3, family.tile, family.tile))
            report["setup"]["first_tile"] = session.get_session_info()
            report["native"]["buffer_policy"] = "preallocated CUDA input and output at the measured shape"
            runners = {1: NativeRunner(session)}
        else:
            runners = torch_runners(checkpoint, family, batch, args.backend, report)
        if reference:
            report["tiny_inputs"] = check_tiny(runners[1], family.scale)
        report["stress"] = validate_stress(runners, paths, family, batch, args.backend, base)
        stress_comparisons = {label: row for label, row in report["stress"].items() if row.get("mse") is not None}
        if stress_comparisons:
            report["worst_stress_psnr"] = max(stress_comparisons, key=lambda label: stress_comparisons[label]["mse"])
        if session is not None:
            shape = (1, 3, family.tile, family.tile)
            session.prepare(shape)
            runners[1].measure(torch.zeros(shape))
            report["setup"]["1"] = session.get_session_info()
        shape_runners: dict[tuple[int, int], Runner] | None = None
        if args.variable_edge_tiles:
            shapes = set()
            tile = family.tile
            for path in paths:
                with Image.open(path) as image:
                    width, height = image.size
                shapes.update(
                    (max(3, min(tile, height - top)), max(3, min(tile, width - left)))
                    for top, left in grid(height, width, tile, family.overlap)
                )
            shape_runners = {}
            setup = report["setup"]["prewarmed_shapes"] = {}
            with NvmlSampler(nvml) as memory:
                for height, width in sorted(shapes, key=lambda hw: (math.prod(hw), hw), reverse=True):
                    shape = (1, 3, height, width)
                    stage: dict[str, Any] = {"input_shape": shape}
                    setup[f"{height}x{width}"] = stage
                    print(f"Prewarming {height}x{width}", flush=True)
                    runner = runners[1]
                    shape_session = session
                    if options is not None and (height, width) != (tile, tile):
                        shape_session = stack.enter_context(InferenceSession(args.native_model, options))
                        started = time.perf_counter()
                        shape_session.prepare(shape)
                        stage["prepare_seconds"] = time.perf_counter() - started
                        runner = NativeRunner(shape_session)
                    stage["warm_call_seconds"] = runner.measure(torch.zeros(shape))[1]
                    if shape_session is not None:
                        stage["native"] = shape_session.get_session_info()
                    shape_runners[height, width] = runner
                    save()
            report["vram"]["shape_prewarm"] = memory.report(background["used_mib"])
            report["vram"]["all_shapes_warm"] = nvml.read()
        save()
        for path in paths:
            image = read_image(path)
            with NvmlSampler(nvml) as memory:
                row, output = timed_page(image, path.name, runners, family, batch, shape_runners=shape_runners)
            row["vram"] = memory.report(background["used_mib"])
            row["vram"]["peak_above_setup_background_mib"] = row["vram"].pop("peak_above_pre_context_background_mib")
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
        stack.close()
        save()
        print(f"{report['status']}: {directory / 'report.json'}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
