"""Compare GateRV3 Vulkan, eager PyTorch, and ONNX Runtime WebGPU on image tiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Event, Thread
from typing import NamedTuple, NotRequired, TypedDict

import numpy as np
import torch
from benchmark import IMAGE_EXTS, ROOT, image_tensor, patches, tensor_image
from PIL import Image
from reference.model import load_model

from umzi_native import GateRV3Model, InferenceSession, SessionOptions, peek_model, vulkan_devices
from umzi_native._loader import find_session_library

TILE = 128
BATCH = 16
CHECKPOINT = ROOT / "models/1x_umzi_digital_decompress_gaterv3_1.pth"

Call = Callable[[torch.Tensor], torch.Tensor]
Runner = Callable[[int], AbstractContextManager[tuple[Call, dict]]]
TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
NUMPY_DTYPES = {"fp32": np.float32, "fp16": np.float16}


class Case(NamedTuple):
    backend: str
    precision: str
    device_index: int = 0

    def label(self, torch_device: str) -> str:
        if self.backend == "torch":
            return f"torch_{torch_device}_{self.precision}"
        if self.backend == "vulkan":
            return f"vulkan_device{self.device_index}_{self.precision}"
        return f"onnxruntime_webgpu_{self.precision}"


class Page(NamedTuple):
    name: str
    height: int
    width: int
    tiles: list[tuple[int, int, torch.Tensor]]


class ImageTiming(TypedDict):
    image: str
    tiles: int
    batch16_calls: int
    source_pixels: int
    batch16_seconds: float
    batch16_megapixels_per_second: float
    batch16_tiles_per_second: float
    batch1_seconds: NotRequired[float]
    batch1_average_tile_latency_ms: NotRequired[float]


class TileTiming(TypedDict):
    image: str
    tile_index: int
    top: int
    left: int
    batch1_tile_latency_seconds: float


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_page(path: Path) -> Page:
    with Image.open(path) as image:
        sample = tensor_image(image)
    return Page(path.name, sample.shape[2], sample.shape[3], patches(sample, TILE))


def load_pages(directory: Path) -> list[Page]:
    pages = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            pages.append(load_page(path))
    if not pages:
        raise ValueError(f"No benchmark images found in {directory}")
    return pages


def torch_call(model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> Call:
    def run(sample: torch.Tensor) -> torch.Tensor:
        return model(sample.to(device=device, dtype=dtype)).cpu().float()

    return run


def create_webgpu(path: Path, batch: int, precision: str) -> tuple[Call, dict]:
    import onnxruntime as ort
    import onnxruntime_ep_webgpu as plugin  # pyrefly: ignore[missing-import]

    name = plugin.get_ep_name()
    ort.disable_telemetry_events()
    devices = [device for device in ort.get_ep_devices() if device.ep_name == name]
    if not devices:
        ort.register_execution_provider_library("umzi_webgpu", plugin.get_library_path())
        devices = [device for device in ort.get_ep_devices() if device.ep_name == name]
    gpu = next((device for device in devices if device.device.type == ort.OrtHardwareDeviceType.GPU), None)
    if gpu is None:
        raise RuntimeError("No ONNX Runtime WebGPU GPU device is available")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    options.add_provider_for_devices([gpu], {"preferredLayout": "NHWC"})
    try:
        session = ort.InferenceSession(str(path), sess_options=options, enable_fallback=False)
    except Exception as error:
        # The extension exposes provider-specific exception classes.
        raise RuntimeError(f"WebGPU session creation failed for batch {batch}: {error}") from error
    assigned: Counter[str] = Counter()
    for subgraph in session.get_provider_graph_assignment_info():
        assigned[subgraph.ep_name] += len(subgraph.get_nodes())
    if name not in session.get_providers() or set(assigned) != {name} or not assigned[name]:
        raise RuntimeError(f"WebGPU-only graph placement was not verified: {dict(assigned)}")
    expected = [batch, 3, TILE, TILE]
    if session.get_inputs()[0].shape != expected or session.get_outputs()[0].shape != expected:
        raise RuntimeError(f"Unexpected input or output shape for {path}")

    dtype = NUMPY_DTYPES[precision]

    def run(sample: torch.Tensor) -> torch.Tensor:
        try:
            (output,) = session.run(["output"], {"input": sample.numpy().astype(dtype, copy=False)})
        except Exception as error:
            raise RuntimeError(f"WebGPU inference failed for batch {batch}: {error}") from error
        return torch.from_numpy(output).float()

    return run, {
        "providers": session.get_providers(),
        "assigned_nodes": dict(assigned),
        "vendor": gpu.device.vendor,
        "device_id": gpu.device.device_id,
        "onnxruntime": ort.__version__,
        "plugin": importlib.metadata.version("onnxruntime-ep-webgpu"),
        "onnx": str(path),
    }


def onnx_path(output: Path, precision: str, batch: int) -> Path:
    return output / f"gaterv3-{precision}-t{TILE}-b{batch}.onnx"


def export_webgpu(checkpoint: Path, output: Path, precision: str) -> None:
    dtype = TORCH_DTYPES[precision]
    model, _ = load_model(checkpoint, device="cpu", dtype=dtype)
    for batch in (1, BATCH):
        torch.onnx.export(
            model,
            (torch.zeros(batch, 3, TILE, TILE, dtype=dtype),),
            str(onnx_path(output, precision, batch)),
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamo=False,
        )


def vulkan_call(session: InferenceSession) -> Call:
    def run(sample: torch.Tensor) -> torch.Tensor:
        (output,) = session.run(["output"], {"input": sample.numpy()})
        return torch.from_numpy(output)

    return run


def batch_sample(tiles: list[tuple[int, int, torch.Tensor]]) -> torch.Tensor:
    samples = [tile[2] for tile in tiles]
    return torch.cat(samples + [samples[-1]] * (BATCH - len(samples)))


def psnr(mse: float) -> float | None:
    return float(-10 * np.log10(mse)) if mse else None


@contextmanager
def open_runner(args: argparse.Namespace, case: Case, batch: int) -> Generator[tuple[Call, dict]]:
    if case.backend == "vulkan":
        options = SessionOptions(
            backend="vulkan",
            device_id=case.device_index,
            precision=case.precision,
            dll_path=args.library,
            kernel_options=args.vulkan_options,
        )
        with InferenceSession(args.model, options) as session:
            session.prepare((batch, 3, TILE, TILE))
            yield vulkan_call(session), session.get_session_info()
    elif case.backend == "webgpu":
        yield create_webgpu(onnx_path(args.output, case.precision, batch), batch, case.precision)
    else:
        device = torch.device(args.torch_device)
        dtype = TORCH_DTYPES[case.precision]
        model, config = load_model(args.checkpoint, device=device, dtype=dtype)
        yield torch_call(model, device, dtype), {"model_config": config.describe()}


@torch.inference_mode()
def memory_worker(connection: Connection, args: argparse.Namespace, case: Case, image: Path, batch: int) -> None:
    try:
        torch.set_num_threads(4)
        tiles = load_page(image).tiles
        stride = batch
        inputs = [
            batch_sample(tiles[i : i + stride]) if batch == BATCH else tiles[i][2]
            for i in range(0, min(len(tiles), 3 * stride), stride)
        ]
        # This wraps Metal's process-wide currentAllocatedSize. It also
        # observes allocations made by the Vulkan and WebGPU Metal drivers.
        allocated = torch.mps.driver_allocated_memory
        baseline = allocated()
        with open_runner(args, case, batch) as (run, _):
            prepared = allocated()
            for _ in range(args.warmup):
                run(inputs[0])
            warm = allocated()
            samples = [warm]
            stop = Event()

            def sample_allocations() -> None:
                while not stop.wait(0.005):
                    samples.append(allocated())

            sampler = Thread(target=sample_allocations)
            sampler.start()
            try:
                for sample in inputs:
                    run(sample)
                    samples.append(allocated())
            finally:
                stop.set()
                sampler.join()
            connection.send({
                "method": "Metal currentAllocatedSize",
                "baseline_bytes": baseline,
                "prepared_delta_bytes": max(0, prepared - baseline),
                "warm_resident_delta_bytes": max(0, warm - baseline),
                "sampled_peak_delta_bytes": max(0, max(samples) - baseline),
                "sample_count": len(samples),
            })
    except (ImportError, MemoryError, OSError, RuntimeError, ValueError) as error:
        connection.send({"error": str(error)})
    finally:
        connection.close()


def measure_memory(args: argparse.Namespace, case: Case, image: Path, batch: int) -> dict:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=memory_worker, args=(sender, args, case, image, batch))
    with receiver, sender:
        process.start()
        sender.close()
        try:
            if not receiver.poll(180):
                return {"error": "GPU memory probe timed out"}
            return receiver.recv()
        except EOFError:
            return {"error": "GPU memory probe exited before returning a sample"}
        finally:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if not process.is_alive():
                process.close()


def evaluate(
    open_call: Runner,
    pages: list[Page],
    output: Path,
    warmup: int,
    reference: dict[str, torch.Tensor],
    reference_dir: Path,
    ssimulacra2: str,
    *,
    is_reference: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    image_rows: list[ImageTiming] = []
    latency_rows: list[TileTiming] = []
    quality_rows: list[dict[str, str | float | None]] = []
    scores = []
    total_squared_error = 0.0
    with open_call(BATCH) as (run_batch, batch_info):
        for _ in range(warmup):
            run_batch(batch_sample(pages[0].tiles[:BATCH]))
        for page in pages:
            result = torch.empty((1, 3, page.height, page.width), dtype=torch.float32)
            elapsed = 0.0
            for start in range(0, len(page.tiles), BATCH):
                group = page.tiles[start : start + BATCH]
                inputs = batch_sample(group)
                started = time.perf_counter()
                restored = run_batch(inputs)
                elapsed += time.perf_counter() - started
                if not torch.isfinite(restored).all():
                    raise RuntimeError(f"Nonfinite batch-{BATCH} output on {page.name}, tile {start}")
                for index, (top, left, _) in enumerate(group):
                    h, w = min(TILE, page.height - top), min(TILE, page.width - left)
                    result[:, :, top : top + h, left : left + w] = restored[index : index + 1, :, :h, :w]
            pixels = page.height * page.width
            image_rows.append({
                "image": page.name,
                "tiles": len(page.tiles),
                "batch16_calls": (len(page.tiles) + BATCH - 1) // BATCH,
                "source_pixels": pixels,
                "batch16_seconds": elapsed,
                "batch16_megapixels_per_second": pixels / elapsed / 1e6,
                "batch16_tiles_per_second": len(page.tiles) / elapsed,
            })

            path = output / f"{page.name}.png"
            image_tensor(result).save(path)
            if is_reference:
                reference[page.name] = result
                quality_rows.append({
                    "image": page.name,
                    "raw_float_sha256": hashlib.sha256(result.numpy().tobytes()).hexdigest(),
                })
            else:
                delta = result.double() - reference[page.name].double()
                squared_error = float(delta.square().sum())
                total_squared_error += squared_error
                score = subprocess.run(
                    [ssimulacra2, str(reference_dir / path.name), str(path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                scores.append(float(score.stdout.strip()))
                quality_rows.append({
                    "image": page.name,
                    "raw_float_psnr_db": psnr(squared_error / result.numel()),
                    "raw_float_max_absolute_error": float(delta.abs().max()),
                    "ssimulacra2": scores[-1],
                })

    del run_batch
    with open_call(1) as (run_single, single_info):
        for _ in range(warmup):
            run_single(pages[0].tiles[0][2])
        for page, row in zip(pages, image_rows, strict=True):
            samples = []
            for index, (top, left, patch) in enumerate(page.tiles):
                started = time.perf_counter()
                restored = run_single(patch)
                seconds = time.perf_counter() - started
                if not torch.isfinite(restored).all():
                    raise RuntimeError(f"Nonfinite batch-1 output on {page.name}, tile {index}")
                samples.append(seconds)
                latency_rows.append({
                    "image": page.name,
                    "tile_index": index,
                    "top": top,
                    "left": left,
                    "batch1_tile_latency_seconds": seconds,
                })
            row["batch1_seconds"] = sum(samples)
            row["batch1_average_tile_latency_ms"] = statistics.mean(samples) * 1000

    seconds = sum(row["batch16_seconds"] for row in image_rows)
    total_pixels = sum(row["source_pixels"] for row in image_rows)
    summary: dict[str, float | None] = {
        "batch16_seconds": seconds,
        "batch16_megapixels_per_second": total_pixels / seconds / 1e6,
        "batch16_tiles_per_second": sum(row["tiles"] for row in image_rows) / seconds,
        "batch1_seconds": sum(row["batch1_tile_latency_seconds"] for row in latency_rows),
        "batch1_average_tile_latency_ms": statistics.mean(row["batch1_tile_latency_seconds"] for row in latency_rows)
        * 1000,
    }
    if not is_reference:
        summary.update({
            "raw_float_psnr_db": psnr(total_squared_error / (3 * total_pixels)),
            "ssimulacra2_mean": statistics.mean(scores),
        })
    return {
        "summary": summary,
        "per_image": image_rows,
        "per_tile": latency_rows,
        "quality": quality_rows,
        "setup": {"batch16": batch_info, "batch1": single_info},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--model", type=Path, help="GateRV3 .ummodel exported from the same checkpoint")
    parser.add_argument("--images", type=Path, default=ROOT / "samples/input")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--devices", action="store_true", help="List Vulkan compute GPUs and exit")
    parser.add_argument("--vulkan-device", type=int, action="append", help="Repeat for multiple Vulkan drivers")
    parser.add_argument("--vulkan-precision", choices=("mixed", "fp32", "fp16"), action="append")
    parser.add_argument("--vulkan-reference", action="store_true", help="Run the original Vulkan shader schedule")
    parser.add_argument(
        "--vulkan-option",
        action="append",
        default=[],
        metavar="NAME=0|1",
        help="Pass a Vulkan kernel option. Repeat to set multiple options.",
    )
    parser.add_argument("--webgpu-precision", choices=("fp32", "fp16"), action="append")
    parser.add_argument("--skip-gpu-memory", action="store_true", help="Skip separate Metal memory runs")
    parser.add_argument("--torch-device", choices=("mps", "cuda"), default="mps")
    parser.add_argument(
        "--backends", nargs="+", choices=("vulkan", "torch", "webgpu"), default=["vulkan", "torch", "webgpu"]
    )
    parser.add_argument("--ssimulacra2", default="ssimulacra2")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if not args.devices and args.output is None:
        parser.error("--output is required for benchmarking")
    if args.warmup < 1:
        parser.error("--warmup must be positive")
    options: dict[str, int] = {}
    for entry in args.vulkan_option:
        name, separator, value = entry.partition("=")
        if not name or not separator or value not in ("0", "1") or name in options:
            parser.error(f"Expected a unique Vulkan option in NAME=0|1 form, got {entry!r}")
        options[name] = int(value)
    if args.vulkan_reference:
        if options.get("reference", 1) != 1:
            parser.error("--vulkan-reference conflicts with reference=0")
        options["reference"] = 1
    args.vulkan_options = options
    if "fp16" in (args.vulkan_precision or ()) and (options.get("reference") or not options.get("fast_convolution", 1)):
        parser.error("Vulkan FP16 arithmetic requires fast_convolution and the optimized schedule")
    if not args.devices and "vulkan" in args.backends and args.model is None:
        parser.error("--model is required for Vulkan")
    return args


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.devices:
        print(json.dumps(vulkan_devices(args.library), indent=2))
        return 0
    torch.set_num_threads(4)
    device = torch.device(args.torch_device)
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK before benchmarking MPS")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tool = shutil.which(args.ssimulacra2)
    if tool is None:
        raise FileNotFoundError(f"SSIMULACRA2 executable not found: {args.ssimulacra2}")
    checkpoint_hash = digest(args.checkpoint)
    if "vulkan" in args.backends:
        payload = peek_model(args.model).payload
        if not isinstance(payload, GateRV3Model) or payload.checkpoint != checkpoint_hash:
            raise ValueError("The GateRV3 .ummodel checkpoint digest does not match --checkpoint")
    pages = load_pages(args.images)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "tile": TILE,
        "throughput_batch": BATCH,
        "batch1_latency_batch": 1,
        "timing_boundary": "CPU tensor to synchronized CPU float32 output; tiling and rendering are excluded",
        "warmup_calls_per_shape": args.warmup,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_device": str(device),
        "checkpoint": {"name": args.checkpoint.name, "sha256": checkpoint_hash},
        "model_package_sha256": digest(args.model) if args.model else None,
        "images": {page.name: digest(args.images / page.name) for page in pages},
        "ssimulacra2": tool,
        "variants": {},
    }
    reports = report["variants"]
    reference_dir = args.output / f"torch_{device.type}_fp32"
    reference: dict[str, torch.Tensor] = {}

    def save() -> None:
        path = args.output / "report.json"
        staging = path.with_suffix(".json.tmp")
        staging.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.replace(path)

    reference_case = Case("torch", "fp32")
    cases = [reference_case]
    if "torch" in args.backends:
        cases.append(Case("torch", "fp16"))
    if "webgpu" in args.backends:
        cases.extend(
            Case("webgpu", precision) for precision in dict.fromkeys(args.webgpu_precision or ["fp32", "fp16"])
        )
    if "vulkan" in args.backends:
        args.library = find_session_library(args.model.resolve(), args.library, backend="vulkan")
        devices = {device["index"]: device for device in vulkan_devices(args.library)}
        report["vulkan_devices"] = list(devices.values())
        report["vulkan_kernel_options"] = args.vulkan_options
        report["vulkan_kernels"] = "reference" if args.vulkan_options.get("reference") else "optimized"
        for index in dict.fromkeys(args.vulkan_device or [0]):
            if index not in devices:
                raise ValueError(f"Unavailable Vulkan device index {index}")
            precisions = ["mixed", "fp32"]
            info = devices[index]
            if (
                info["shader_float16"]
                and not args.vulkan_options.get("reference")
                and args.vulkan_options.get("fast_convolution", 1)
            ):
                precisions.append("fp16")
            cases.extend(
                Case("vulkan", precision, index) for precision in dict.fromkeys(args.vulkan_precision or precisions)
            )

    metal = sys.platform == "darwin" and device.type == "mps"
    first_image = args.images / pages[0].name
    for case in cases:
        name = case.label(device.type)
        is_reference = case == reference_case
        print(f"Running {name}", flush=True)
        try:
            if case.backend == "webgpu":
                export_webgpu(args.checkpoint, args.output, case.precision)
            result = evaluate(
                lambda batch, case=case: open_runner(args, case, batch),
                pages,
                args.output / name,
                args.warmup,
                reference,
                reference_dir,
                tool,
                is_reference=is_reference,
            )
        except (ImportError, MemoryError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            if is_reference:
                raise
            print(f"{name} failed: {error}", file=sys.stderr, flush=True)
            reports[name] = {"error": str(error)}
            save()
            continue
        reports[name] = result
        save()
        print(f"{name} {result['summary']}", flush=True)
        if metal and not args.skip_gpu_memory:
            print(f"Sampling GPU memory for {name}", flush=True)
            result["gpu_memory"] = {
                f"batch{batch}": measure_memory(args, case, first_image, batch) for batch in (1, BATCH)
            }
        else:
            result["gpu_memory"] = {"unavailable": "Metal GPU memory probe was skipped or is unavailable"}
        save()
    return int(any("error" in result for result in reports.values()))


if __name__ == "__main__":
    sys.exit(main())
