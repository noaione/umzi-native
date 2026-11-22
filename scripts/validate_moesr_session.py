"""Validate the packaged MoESR mixed and FP16 sessions against eager CUDA FP32."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from benchmark import image_tensor
from benchmark_architectures import (
    ROOT,
    Runner,
    comparison,
    digest,
    read_image,
    score_stress_render,
    ssimulacra2,
    stress_inputs,
    summarize,
    timed_page,
)
from check_python_session import cli_output
from reference.native_session import DeviceSession
from resselt.archs.moesr.arch import MoESR
from torch import Tensor
from validate_moesr import BASELINE, FAMILY

from umzi_native import InferenceSession, MoESRModel, SessionOptions, peek_model
from umzi_native._abi import ShapeABI, check_status
from umzi_native._model import MOESR_ARCHITECTURE, MOESR_CONFIG, MOESR_FAMILY
from umzi_native.exporter import export_model
from umzi_native.exporter._loader import load_checkpoint
from umzi_native.exporter._package import write_package

PRECISIONS = {"mixed": torch.float32, "fp16": torch.float16}


def verify_packages(checkpoint: Path, model: Path, library: Path) -> dict[str, Any]:
    def rejects(path: Path, label: str) -> None:
        if peek_model(path).supported:
            raise AssertionError(f"MoESR reader accepted {path.name}")
        try:
            with InferenceSession(path, SessionOptions(dll_path=library)):
                raise AssertionError(f"MoESR runtime accepted {path.name}")
        except RuntimeError as error:
            if label not in str(error):
                raise

    info = export_model(checkpoint, model, kernel="moesr")
    if not isinstance(info.payload, MoESRModel) or info.payload.checkpoint != digest(checkpoint):
        raise ValueError("The package does not identify the validation checkpoint")
    with TemporaryDirectory(dir=model.parent) as directory:
        temporary = Path(directory)
        repeat = export_model(checkpoint, temporary / "repeat.ummodel")
        if digest(repeat.path) != digest(model):
            raise AssertionError("MoESR export is nondeterministic")
        architecture = MOESR_ARCHITECTURE.pack(*MOESR_CONFIG)
        for schema, metadata, message in (
            (2, architecture, "payload version"),
            (1, bytes(len(architecture)), "model architecture"),
            (1, architecture, "weight layout"),
        ):
            path = temporary / "invalid.ummodel"
            write_package(path, MOESR_FAMILY, schema, {1: metadata, 3: b"short", 4: bytes(32)})
            rejects(path, message)
    return {"sha256": digest(model), "bytes": model.stat().st_size, "deterministic": True, "payload_rejections": 3}


def check_io(session: InferenceSession, sample: Tensor, output: Tensor) -> dict[str, Any]:
    dims = ShapeABI.from_dimensions(sample.shape)
    dll, handle = session._dll, session._handle
    # The C ABI must reject an output one byte shorter than the 4x result.
    try:
        check_status(
            dll,
            dll.umzi_session_enqueue(
                handle,
                ctypes.byref(dims),
                sample.data_ptr(),
                sample.nbytes,
                output.data_ptr(),
                output.nbytes - 1,
                torch.cuda.current_stream().cuda_stream,
            ),
        )
    except RuntimeError as error:
        if "Contiguous input and output require" not in str(error) or str(output.nbytes) not in str(error):
            raise
    else:
        raise AssertionError("MoESR accepted the undersized output buffer")
    (host,) = session.run(None, {session.get_inputs()[0].name: sample.cpu().numpy()})
    if not np.array_equal(output.cpu().numpy(), host):
        raise AssertionError("MoESR host and device outputs differ")
    return {"short_output_rejected": True, "host_bitwise_match": True}


def check_plan_cache(model: Path, library: Path, precision: str, root: Path) -> dict[str, Any]:
    """Check a cold miss, a warm hit without timing, and key separation by workspace cap."""
    sample = torch.rand((2, 3, 64, 96), generator=torch.Generator().manual_seed(7))
    sample = sample.to(device="cuda", dtype=PRECISIONS[precision])
    cache = root.parent / f"cache-check-{precision}"
    shutil.rmtree(cache, ignore_errors=True)

    def run(expected: str, timed: bool, **kernel_options: int) -> Tensor:
        options = SessionOptions(precision=precision, dll_path=library, cache_dir=cache, kernel_options=kernel_options)
        with InferenceSession(model, options) as session:
            result = DeviceSession(session)(sample)
            torch.cuda.synchronize()
            info = session.get_session_info()
        if info["cache_status"] != expected or bool(info["diagnostics"]["timed_plans"]) != timed:
            raise AssertionError(f"MoESR {precision} plan cache: expected {expected}, got {info['cache_status']}")
        return result

    cold = run("miss", True)
    warm = run("hit", False)
    if not torch.equal(cold, warm):
        raise AssertionError(f"MoESR {precision} output changed after reusing cached plans")
    run("miss", True, workspace_mib=64)
    files = sorted(path.name for path in cache.glob("moesr_*.umplan"))
    if len(files) != 2:
        raise AssertionError(f"MoESR {precision} cache keys did not separate workspace caps: {files}")
    return {"cold": "miss", "warm": "hit", "warm_timed_plans": 0, "warm_bitwise_match": True, "files": len(files)}


def validate_cases(
    graph: InferenceSession,
    direct: InferenceSession,
    cases: dict[str, Tensor],
    expected: dict[str, Tensor],
    dtype: torch.dtype,
    output: Path,
    cli: Path | None,
    model: Path,
    precision: str,
    plans: Path,
) -> dict[str, Any]:
    run, uncaptured = DeviceSession(graph), DeviceSession(direct)
    rows: dict[str, Any] = {}
    alternate = torch.cuda.Stream()
    for name, host in cases.items():
        sample = host.to(device="cuda", dtype=dtype)
        actual = run(sample)
        reference = expected[name]
        if actual.shape != reference.shape or actual.dtype != dtype:
            raise AssertionError(f"MoESR {precision} {name}: wrong dtype or output shape")
        # Timing can pick different plans for near-ties, so the graph session
        # tunes and saves, and the direct session must reuse the same plans.
        if not torch.equal(actual, uncaptured(sample)):
            raise AssertionError(f"MoESR {precision} {name}: graph and direct differ")
        if direct.get_session_info()["cache_status"] != "hit":
            raise AssertionError(f"MoESR {precision} {name}: direct session did not reuse the saved plans")
        row = comparison(actual.float().cpu().numpy(), reference.numpy())
        if not row["finite"]:
            raise AssertionError(f"MoESR {precision} {name}: non-finite output")
        row.update(input_shape=list(sample.shape), output_shape=list(actual.shape), direct_bitwise_match=True)
        if name in ("gray", "gray-128", "checkerboard", "noise"):
            row["ssimulacra2"] = score_stress_render(
                output / "stress" / precision, name, reference, actual.float().cpu()
            )
        if name == "rect-65x79":
            # Reuse the same session on a different CUDA stream.
            with torch.cuda.stream(alternate):
                alternate.wait_stream(torch.cuda.default_stream())
                second = run(sample)
            alternate.synchronize()
            if not torch.equal(second, actual):
                raise AssertionError("MoESR changed output on an alternate stream")
            row.update(alternate_stream_bitwise_match=True, **check_io(graph, sample, actual))
            if cli is not None:
                with TemporaryDirectory(dir=output) as directory:
                    temporary = Path(directory)
                    from_cli = cli_output(
                        cli, model, precision, plans, sample.cpu().numpy(), temporary, tuple(actual.shape)
                    )
                if not np.array_equal(from_cli, actual.cpu().numpy()):
                    raise AssertionError("MoESR CLI and Python outputs differ")
                row["cli_bitwise_match"] = True
        if sample.size(0) > 1:
            row["vs_individual"] = []
            for part, tile in zip(actual.split(1), sample.split(1), strict=True):
                individual = run(tile)
                consistency = comparison(part.float().cpu().numpy(), individual.float().cpu().numpy())
                if not consistency["finite"]:
                    raise AssertionError(f"MoESR {precision} {name}: non-finite single-tile comparison")
                row["vs_individual"].append(consistency)
        rows[name] = row
        print(f"{precision} {name}: {row['psnr_db']:.3f} dB, max {row['max_abs']:.6g}", flush=True)
    for shape in ((1, 3, 1, 1), (1, 3, 1, 2), (1, 3, 2, 1)):
        try:
            graph.prepare(shape)
        except RuntimeError as error:
            if "reflection padding" not in str(error):
                raise
        else:
            raise AssertionError(f"MoESR accepted invalid reflection padding {shape}")
    return rows


def benchmark_pages(
    model: Path, options: SessionOptions, dtype: torch.dtype, sources: list[Path], reference: Path, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    pages, setup = [], {}
    with ExitStack() as stack:
        runners = {}
        for batch in (1, FAMILY.batch):
            session = stack.enter_context(InferenceSession(model, options))
            shape = (batch, 3, FAMILY.tile, FAMILY.tile)
            session.prepare(shape)
            runner = Runner(DeviceSession(session), dtype)
            runner.measure(torch.zeros(shape))
            runners[batch] = runner
            setup[str(batch)] = session.get_session_info()
        for path in sources:
            row, candidate = timed_page(read_image(path), path.name, runners, FAMILY, FAMILY.batch)
            row.update(comparison(candidate.numpy(), np.load(reference / f"{path.stem}.npy", mmap_mode="r")))
            if not row["finite"]:
                raise AssertionError(f"Non-finite MoESR output on {path.name}")
            rendered = output / path.name
            image_tensor(candidate).save(rendered)
            row["ssimulacra2"] = ssimulacra2(reference / path.name, rendered)
            pages.append(row)
            print(
                f"{options.precision} {path.name}: {row['psnr_db']:.3f} dB, SSIMULACRA2 {row['ssimulacra2']:.3f}",
                flush=True,
            )
    return {"setup": setup, "pages": pages, "summary": summarize(pages)}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output/moesr-session/validated")
    parser.add_argument("--reference-report", type=Path, default=BASELINE / "report.json")
    parser.add_argument("--pages", action="store_true")
    args = parser.parse_args()
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "models" / FAMILY.checkpoint
    sources = sorted((ROOT / "samples" / FAMILY.samples).glob("*.png"))
    config = {
        "checkpoint_sha256": digest(checkpoint),
        "images": {path.name: digest(path) for path in sources},
        "tile": FAMILY.tile,
        "batch": FAMILY.batch,
        "overlap": FAMILY.overlap,
        "scale": FAMILY.scale,
    }
    saved = json.loads(args.reference_report.read_text(encoding="utf-8"))
    if (
        saved["backend"] != "fp32"
        or saved["status"] != "complete"
        or any(saved["config"][key] != value for key, value in config.items())
    ):
        raise ValueError("The eager FP32 report does not match the MoESR checkpoint, images, or tiling")
    model = args.output / "moesr.ummodel"
    report: dict[str, Any] = {
        "config": config,
        "package": verify_packages(checkpoint, model, args.library),
        "library_sha256": digest(args.library),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": {},
    }
    print("MoESR package checks passed", flush=True)
    reference, _ = load_checkpoint(checkpoint)
    if not isinstance(reference, MoESR):
        raise TypeError("The validation checkpoint must load as MoESR")
    reference.eval().cuda()
    cases = dict(stress_inputs(FAMILY.tile, sources, FAMILY.batch))
    for height, width in ((2, 2), (3, 5), (8, 9)):
        cases[f"tiny-{height}x{width}"] = torch.zeros((1, 3, height, width))
    expected = {key: reference(value.cuda()).cpu() for key, value in cases.items()}
    del reference
    for precision, dtype in PRECISIONS.items():
        plans = args.output / "plans" / precision
        shutil.rmtree(plans, ignore_errors=True)
        options = SessionOptions(precision=precision, dll_path=args.library, cache_dir=plans)
        direct_options = SessionOptions(
            precision=precision, dll_path=args.library, enable_cuda_graph=False, cache_dir=plans
        )
        with InferenceSession(model, options) as graph, InferenceSession(model, direct_options) as direct:
            stress = validate_cases(
                graph, direct, cases, expected, dtype, args.output, args.cli, model, precision, plans
            )
            stress["plan_cache"] = check_plan_cache(model, args.library, precision, plans)
        report["results"][precision] = {"stress": stress}

    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.pages:
        for precision, dtype in PRECISIONS.items():
            options = SessionOptions(precision=precision, dll_path=args.library)
            report["results"][precision].update(
                benchmark_pages(model, options, dtype, sources, args.reference_report.parent, args.output / precision)
            )
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
