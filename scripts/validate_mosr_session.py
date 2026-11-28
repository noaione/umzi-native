"""Validate the packaged MoSR FP32, mixed, and FP16 sessions against eager CUDA FP32."""

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
from benchmark_architectures import FAMILIES, ROOT, comparison, digest, read_image
from check_python_session import cli_output
from reference.native_session import DeviceSession
from torch import Tensor

from umzi_native import InferenceSession, MoSRModel, SessionOptions, peek_model
from umzi_native._abi import ShapeABI, check_status
from umzi_native._model import MOSR_ARCHITECTURE, MOSR_FAMILY
from umzi_native.exporter import export_model
from umzi_native.exporter._loader import load_checkpoint
from umzi_native.exporter._package import write_package

PRECISIONS = {"fp32": torch.float32, "mixed": torch.float32, "fp16": torch.float16}


def verify_packages(checkpoint: Path, model: Path, library: Path) -> dict[str, Any]:
    def rejects(path: Path, label: str) -> None:
        if peek_model(path).supported:
            raise AssertionError(f"MoSR reader accepted {path.name}")
        try:
            with InferenceSession(path, SessionOptions(dll_path=library)):
                raise AssertionError(f"MoSR runtime accepted {path.name}")
        except RuntimeError as error:
            if label not in str(error):
                raise

    info = export_model(checkpoint, model, kernel="mosr")
    if not isinstance(info.payload, MoSRModel) or info.payload.checkpoint != digest(checkpoint):
        raise ValueError("The package does not identify the validation checkpoint")
    with TemporaryDirectory(dir=model.parent) as directory:
        temporary = Path(directory)
        repeat = export_model(checkpoint, temporary / "repeat.ummodel")
        if digest(repeat.path) != digest(model):
            raise AssertionError("MoSR export is nondeterministic")
        architecture = MOSR_ARCHITECTURE.pack(
            info.payload.input_channels,
            info.payload.output_channels,
            info.payload.scale,
            info.payload.blocks,
            info.payload.dim,
            info.payload.hidden,
            info.payload.kernel_size,
            {"ps": 0, "gps": 1, "dys": 2}[info.payload.upsampler],
            info.payload.dysample_groups,
            info.payload.flags,
        )
        for schema, metadata, message in (
            (2, architecture, "payload version"),
            (1, bytes(len(architecture)), "model architecture"),
            (1, architecture, "weight layout"),
        ):
            path = temporary / "invalid.ummodel"
            write_package(path, MOSR_FAMILY, schema, {1: metadata, 3: b"short", 4: bytes(32)})
            rejects(path, message)
    return {
        "sha256": digest(model),
        "bytes": model.stat().st_size,
        "deterministic": True,
        "payload_rejections": 3,
        "payload": {
            "scale": info.payload.scale,
            "blocks": info.payload.blocks,
            "dim": info.payload.dim,
            "hidden": info.payload.hidden,
            "kernel_size": info.payload.kernel_size,
            "upsampler": info.payload.upsampler,
            "dysample_groups": info.payload.dysample_groups,
        },
    }


def check_io(session: InferenceSession, sample: Tensor, output: Tensor) -> dict[str, Any]:
    dims = ShapeABI.from_dimensions(sample.shape)
    dll, handle = session._dll, session._handle
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
        raise AssertionError("MoSR accepted an undersized output buffer")
    (host,) = session.run(None, {session.get_inputs()[0].name: sample.cpu().numpy()})
    if not np.array_equal(output.cpu().numpy(), host):
        raise AssertionError("MoSR host and device outputs differ")
    return {"short_output_rejected": True, "host_bitwise_match": True}


def check_plan_cache(model: Path, library: Path, precision: str, root: Path) -> dict[str, Any]:
    sample = torch.rand((1, 3, 64, 96), generator=torch.Generator().manual_seed(7))
    sample = sample.to(device="cuda", dtype=PRECISIONS[precision])
    cache = root / f"cache-check-{precision}"
    shutil.rmtree(cache, ignore_errors=True)

    def run(
        expected: str, timed: bool, cache_dir: Path = cache, **kernel_options: int
    ) -> tuple[Tensor, dict[str, Any]]:
        options = SessionOptions(
            precision=precision, dll_path=library, cache_dir=cache_dir, kernel_options=kernel_options
        )
        with InferenceSession(model, options) as session:
            result = DeviceSession(session)(sample)
            torch.cuda.synchronize()
            info = session.get_session_info()
            assert info["cache_status"] == expected, f"Expected {expected}, got {info}"
            diag = info["diagnostics"]
            if timed:
                assert diag["timed_plans"] > 0, diag
            else:
                assert diag["timed_plans"] == 0, diag
            return result, info

    cold, _ = run("miss", True)
    warm, _ = run("hit", False)
    if not torch.equal(cold, warm):
        raise AssertionError("Plan cache hit changed MoSR output")
    files = list(cache.glob("mosr_*.umplan"))
    assert len(files) == 1, files
    ws0_cold, _ = run("miss", True, workspace_mib=0)
    ws0_warm, _ = run("hit", False, workspace_mib=0)
    assert torch.equal(ws0_cold, ws0_warm)
    assert len(list(cache.glob("mosr_*.umplan"))) == 2
    files[0].write_bytes(b"corrupt")
    rebuilt, _ = run("rebuilt", True)
    rebuilt_hit, _ = run("hit", False)
    assert torch.equal(rebuilt, rebuilt_hit)
    bad_dir = root / "bad-cache-file"
    bad_dir.write_text("occupied", encoding="utf-8")
    _, fallback = run("unavailable", True, cache_dir=bad_dir)
    assert fallback["cache_message"]
    return {"cold_warm_bitwise": True, "workspace_key_isolated": True, "corrupt_rebuilt": True}


def check_options(model: Path, library: Path) -> dict[str, Any]:
    bad_cases = (
        {"unknown_option": 1},
        {"workspace_mib": -1},
        {"workspace_mib": 4097},
    )
    for bad in bad_cases:
        try:
            with InferenceSession(model, SessionOptions(dll_path=library, kernel_options=bad)):
                raise AssertionError(f"Accepted invalid MoSR options: {bad}")
        except RuntimeError:
            pass
    return {"invalid_options_rejected": len(bad_cases)}


def validate_family(
    name: str,
    library: Path,
    cli: Path | None,
    output_dir: Path,
    *,
    run_cache: bool = False,
) -> dict[str, Any]:
    family = FAMILIES[name]
    checkpoint = ROOT / "models" / family.checkpoint
    model_path = output_dir / f"{name}.ummodel"
    pkg_info = verify_packages(checkpoint, model_path, library)
    ref_model, _ = load_checkpoint(checkpoint)
    ref_model.eval().cuda()

    paths = sorted((ROOT / "samples" / family.samples).glob("*.jpg"))
    image_path = paths[0]
    crop = read_image(image_path)[:, :, :65, :79].contiguous()
    tiny = torch.rand((1, 3, 1, 1), generator=torch.Generator().manual_seed(11))
    with torch.inference_mode():
        expected_crop = ref_model(crop.cuda()).cpu().numpy()
        expected_tiny = ref_model(tiny.cuda()).cpu().numpy()

    results: dict[str, Any] = {"package": pkg_info, "precisions": {}}
    for precision in ("fp32", "mixed", "fp16"):
        cache = output_dir / "plans" / name / precision
        options = SessionOptions(precision=precision, dll_path=library, cache_dir=cache)
        direct_options = SessionOptions(precision=precision, dll_path=library, cache_dir=cache, enable_cuda_graph=False)
        with ExitStack() as stack:
            graph = stack.enter_context(InferenceSession(model_path, options))
            direct = stack.enter_context(InferenceSession(model_path, direct_options))
            assert (graph.kernel, graph.scale_factor, graph.context) == ("mosr", family.scale, 0)
            captured, uncaptured = DeviceSession(graph), DeviceSession(direct)
            sample = crop.to(device="cuda", dtype=captured.dtype)
            out = captured(sample)
            torch.cuda.synchronize()
            actual = out.cpu().numpy()
            io_info = check_io(graph, sample, out)
            if not np.array_equal(uncaptured(sample).cpu().numpy(), actual):
                raise AssertionError(f"{name} {precision}: graph and direct differ")
            alternate = torch.cuda.Stream()
            with torch.cuda.stream(alternate):
                alternate.wait_stream(torch.cuda.default_stream())
                alt_out = captured(sample)
            alternate.synchronize()
            if not np.array_equal(alt_out.cpu().numpy(), actual):
                raise AssertionError(f"{name} {precision}: alternate stream differs")
            metrics = comparison(actual, expected_crop)
            min_psnr = 100.0 if precision == "fp32" else (55.0 if precision == "mixed" else 48.0)
            if not metrics["finite"] or (metrics["psnr_db"] is not None and metrics["psnr_db"] < min_psnr):
                raise AssertionError(f"Poor {name} {precision} parity on {image_path.name}: {metrics}")
            if cli is not None and name == "mosr":
                cli_tmp = output_dir / "cli-tmp"
                cli_tmp.mkdir(parents=True, exist_ok=True)
                cli_res = cli_output(
                    cli,
                    model_path,
                    precision,
                    cache,
                    sample.cpu().numpy(),
                    cli_tmp,
                    output_shape=expected_crop.shape,
                )
                if not np.array_equal(cli_res, actual):
                    raise AssertionError(f"{name} {precision}: CLI differs from Python host")
            tiny_sample = tiny.to(device="cuda", dtype=captured.dtype)
            tiny_metrics = comparison(captured(tiny_sample).cpu().numpy(), expected_tiny)
            if not tiny_metrics["finite"]:
                raise AssertionError(f"{name} {precision}: non-finite tiny 1x1 output")
            entry: dict[str, Any] = {
                "crop_metrics": metrics,
                "tiny_1x1_metrics": tiny_metrics,
                "io": io_info,
                "session": graph.get_session_info(),
            }
            if run_cache and precision == "mixed":
                entry["plan_cache"] = check_plan_cache(model_path, library, precision, output_dir)
                entry["options"] = check_options(model_path, library)
            results["precisions"][precision] = entry
            print(f"[{name}] {precision}: crop={metrics}, tiny={tiny_metrics}", flush=True)
    return results


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output/mosr-session/validation")
    parser.add_argument("--families", nargs="+", default=["mosr", "mosr-l", "mosr-t"])
    args = parser.parse_args()
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"library": str(args.library.resolve()), "families": {}}
    for idx, name in enumerate(args.families):
        report["families"][name] = validate_family(name, args.library, args.cli, args.output, run_cache=(idx == 0))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation passed: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
