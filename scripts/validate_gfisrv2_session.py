"""Validate the packaged GFISRv2 FP32 session against untouched eager FP32."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import numpy as np
import torch
from benchmark import image_tensor
from benchmark_architectures import (
    ROOT,
    Family,
    Runner,
    digest,
    read_image,
    score_stress_render,
    ssimulacra2,
    stress_inputs,
    summarize,
    timed_page,
)
from check_python_session import cli_output
from PIL import Image, ImageChops
from reference.native_session import DeviceSession
from resselt.archs.gfisrv2.arch import GFISRV2
from validate_gfisrv2 import BASELINE, CUFFT_TOLERANCE, FAMILY, compare_stage, raw_metrics

from umzi_native import GFISRv2Model, InferenceSession, SessionOptions, peek_model
from umzi_native._model import GFISRV2_ARCHITECTURE, GFISRV2_CONFIG, GFISRV2_FAMILY
from umzi_native.exporter import export_model
from umzi_native.exporter._loader import load_checkpoint
from umzi_native.exporter._package import write_package


def export_and_check(checkpoint: Path, package: Path, library: Path) -> dict[str, Any]:
    info = export_model(checkpoint, package, kernel="gfisrv2")
    if not isinstance(info.payload, GFISRv2Model) or info.payload.checkpoint != digest(checkpoint):
        raise AssertionError("GFISRv2 export identifies a different checkpoint")
    rejections: dict[str, str] = {}
    with TemporaryDirectory(dir=package.parent) as directory:
        temporary = Path(directory)
        repeat = export_model(checkpoint, temporary / "repeat.ummodel")
        if digest(repeat.path) != digest(package):
            raise AssertionError("GFISRv2 export was not deterministic")
        metadata = GFISRV2_ARCHITECTURE.pack(*GFISRV2_CONFIG)
        for schema, architecture, message in (
            (2, metadata, "payload version"),
            (1, bytes(len(metadata)), "model architecture"),
            (1, metadata, "weight layout"),
        ):
            invalid = temporary / "invalid.ummodel"
            write_package(invalid, GFISRV2_FAMILY, schema, {1: architecture, 3: b"short", 4: bytes(32)})
            if peek_model(invalid).supported:
                raise AssertionError(f"Python accepted invalid GFISRv2 {message}")
            try:
                with InferenceSession(invalid, SessionOptions(precision="fp32", dll_path=library)):
                    raise AssertionError(f"C++ accepted invalid GFISRv2 {message}")
            except RuntimeError as error:
                if message not in str(error):
                    raise
                rejections[message] = str(error)
    for name, value, message in (
        ("unexact_erf", -1, "unexact_erf must be 0 or 1"),
        ("unexact_erf", 2, "unexact_erf must be 0 or 1"),
        ("unknown", 1, "Unknown GFISRv2 option"),
    ):
        try:
            with InferenceSession(package, SessionOptions(dll_path=library, kernel_options={name: value})):
                raise AssertionError(f"GFISRv2 accepted {name}={value}")
        except RuntimeError as error:
            if message not in str(error):
                raise
            rejections[f"{name}={value}"] = str(error)
    return {"sha256": digest(package), "bytes": package.stat().st_size, "deterministic": True, "rejections": rejections}


def check_stress(
    graph: InferenceSession,
    direct: InferenceSession,
    eager: GFISRV2,
    eager_half: GFISRV2 | None,
    sources: list[Path],
    output: Path,
    settings: SessionOptions,
) -> dict[str, Any]:
    runner, uncaptured = DeviceSession(graph), DeviceSession(direct)
    check_fp32_parity = settings.precision == "fp32" and not settings.kernel_options.get("unexact_erf")
    cases = dict(stress_inputs(FAMILY.tile, sources, FAMILY.batch))
    for height, width in ((1, 1), (1, 7), (2, 2), (3, 5), (8, 9)):
        cases[f"tiny-{height}x{width}"] = torch.full((1, 3, height, width), 0.5)
    results = {}
    for name, host in cases.items():
        reference_input = host.cuda()
        source = reference_input.to(runner.dtype)
        expected, actual = eager(reference_input).cpu(), runner(source).cpu().float()
        row = (
            compare_stage(name, actual, expected)
            if check_fp32_parity
            else raw_metrics(actual.numpy(), expected.numpy())
        )
        if not row["finite"]:
            raise AssertionError(f"Non-finite GFISRv2 output for {name}")
        if not torch.equal(actual, uncaptured(source).cpu().float()):
            raise AssertionError(f"GFISRv2 graph and direct output differ for {name}")
        row.update(input_shape=list(source.shape), output_shape=list(actual.shape), graph_direct_match=True)
        half_expected = eager_half(source).cpu().float() if eager_half is not None else None
        if half_expected is not None:
            row["vs_eager_fp16"] = raw_metrics(actual.numpy(), half_expected.numpy())
        if name in ("gray", "gray-128", "checkerboard", "noise"):
            row["ssimulacra2"] = score_stress_render(output / "stress-renders", name, expected, actual)
        if name in ("gray", "gray-128"):
            directory = output / "stress-raw" / name
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "input.npy", host.numpy())
            np.save(directory / "eager-fp32.npy", expected.numpy())
            np.save(directory / f"native-{settings.precision}.npy", actual.numpy())
            if half_expected is not None:
                np.save(directory / "eager-fp16.npy", half_expected.numpy())
            row["raw_outputs"] = directory.relative_to(output).as_posix()
            values = actual.numpy()
            level = float(host[0, 0, 0, 0])
            row["flat_input"] = {
                "level": level,
                "output_min": float(values.min()),
                "output_max": float(values.max()),
                "output_std": float(values.std(dtype=np.float64)),
                "output_rmse_from_input_level": float(np.sqrt(np.mean((values - level) ** 2, dtype=np.float64))),
                "output_outside_0_1_fraction": float(np.mean((values < 0) | (values > 1))),
            }
        if source.size(0) > 1:
            row["vs_individual"] = [
                compare_stage(f"{name} tile {i}", item, runner(tile).cpu().float())
                if check_fp32_parity
                else raw_metrics(item.numpy(), runner(tile).cpu().float().numpy())
                for i, (item, tile) in enumerate(zip(actual.split(1), source.split(1), strict=True))
            ]
        results[name] = row
        print(f"{name}: max absolute error {row['max_abs']:.6g}", flush=True)
    return results


def check_dispatch(
    graph: InferenceSession, model: Path, cli: Path | None, output: Path, settings: SessionOptions
) -> dict[str, bool]:
    generator = torch.Generator().manual_seed(717)
    device_runner = DeviceSession(graph)
    sample = torch.rand((2, 3, 31, 45), generator=generator).to(device_runner.dtype)
    (host,) = graph.run(None, {"input": sample.numpy()})
    source = sample.cuda()
    alternate = torch.cuda.Stream()
    alternate.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(alternate):
        device = device_runner(source)
    alternate.synchronize()
    if host.shape != (2, 3, 124, 180) or not np.array_equal(host, device.cpu().numpy()):
        raise AssertionError("GFISRv2 host and alternate CUDA stream outputs differ")
    unexact = settings.kernel_options.get("unexact_erf", 0)
    if graph.get_session_info()["diagnostics"]["unexact_erf"] != unexact:
        raise AssertionError("GFISRv2 reported a different erf mode")
    results = {"host_device_alternate_stream_match": True}
    if cli is not None:
        with TemporaryDirectory(dir=output) as directory:
            temporary = Path(directory)
            switches = ("--unexact-erf",) if unexact else ()
            plans = Path(settings.cache_dir) if settings.cache_dir is not None else temporary / "plans"
            reference = cli_output(
                cli, model, settings.precision, plans, sample.numpy(), temporary, host.shape, switches
            )
        if not np.array_equal(host, reference):
            raise AssertionError("GFISRv2 CLI and Python outputs differ")
        results["cli_host_match"] = True
    return results


def check_pages(
    package: Path,
    settings: SessionOptions,
    sources: list[Path],
    reference: Path,
    output: Path,
    check_fp32_parity: bool,
    eager_half_reference: Path | None,
    family: Family,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, setup = [], {}
    torch.cuda.empty_cache()
    with ExitStack() as stack:
        runners = {}
        for batch in (1, family.batch):
            free_before = torch.cuda.mem_get_info()[0]
            session = stack.enter_context(InferenceSession(package, settings))
            shape = (batch, 3, family.tile, family.tile)
            session.prepare(shape)
            device_runner = DeviceSession(session)
            runner = Runner(device_runner, device_runner.dtype)
            runner.measure(torch.zeros(shape))
            runners[batch] = runner
            setup[str(batch)] = session.get_session_info()
            torch.cuda.empty_cache()
            setup[str(batch)]["warmed_device_vram_delta_bytes"] = free_before - torch.cuda.mem_get_info()[0]
        for path in sources:
            row, actual = timed_page(read_image(path), path.name, runners, family, family.batch)
            expected = np.load(reference / f"{path.stem}.npy", mmap_mode="r")
            if check_fp32_parity:
                np.testing.assert_allclose(actual.numpy(), expected, err_msg=path.name, **CUFFT_TOLERANCE)
            row.update(raw_metrics(actual.numpy(), expected))
            if not row["finite"]:
                raise AssertionError(f"Non-finite GFISRv2 output for {path.name}")
            rendered = output / f"{path.stem}-session-{settings.precision}.png"
            image = image_tensor(actual)
            image.save(rendered)
            row["ssimulacra2"] = ssimulacra2(reference / path.name, rendered)
            if eager_half_reference is not None:
                with Image.open(eager_half_reference / path.name) as half_image:
                    row["eager_fp16_png_match"] = (
                        ImageChops.difference(image, half_image.convert("RGB")).getbbox() is None
                    )
            print(f"{path.name}: max error {row['max_abs']:.6g}, SSIMULACRA2 {row['ssimulacra2']:.3f}", flush=True)
            rows.append(row)
    return rows, setup


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-report", type=Path, default=BASELINE / "report.json")
    parser.add_argument("--pages", action="store_true")
    parser.add_argument("--overlap", type=int, default=FAMILY.overlap, help="Tile overlap in input pixels")
    parser.add_argument("--precision", choices=("fp32", "fp16", "mixed"), default="fp32")
    parser.add_argument(
        "--unexact-erf", action="store_true", help="Measure CUDA erff without asserting reference parity"
    )
    args = parser.parse_args()
    if args.overlap < 0 or args.overlap >= FAMILY.tile:
        parser.error(f"--overlap must satisfy 0 <= overlap < {FAMILY.tile}")
    family = replace(FAMILY, overlap=args.overlap)
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint = ROOT / "models" / family.checkpoint
    sources = sorted((ROOT / "samples" / family.samples).glob("*.png"))
    config = {
        "checkpoint_sha256": digest(checkpoint),
        "images": {path.name: digest(path) for path in sources},
        "tile": family.tile,
        "batch": family.batch,
        "overlap": family.overlap,
        "scale": family.scale,
    }
    eager_half_reference = None
    if args.pages:
        ref = json.loads(args.reference_report.read_text(encoding="utf-8"))
        if (
            ref["backend"] != "fp32"
            or ref["status"] != "complete"
            or any(ref["config"][key] != value for key, value in config.items())
        ):
            raise ValueError("The saved FP32 report has a different checkpoint, sample or tiling configuration")
        half_report = args.reference_report.parent.parent / "eager-fp16/report.json"
        if args.precision == "fp16" and half_report.is_file():
            half_ref = json.loads(half_report.read_text(encoding="utf-8"))
            if (
                half_ref["backend"] != "eager-fp16"
                or half_ref["status"] != "complete"
                or any(half_ref["config"][key] != value for key, value in config.items())
            ):
                raise ValueError(
                    "The saved eager FP16 report has a different checkpoint, sample or tiling configuration"
                )
            eager_half_reference = half_report.parent
    if args.output is None:
        directory = "unexact-erf" if args.unexact_erf else "validated"
        if args.precision != "fp32":
            directory += f"-{args.precision}"
        args.output = ROOT / "output/gfisrv2-session" / directory
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    package = args.output / "gfisrv2.ummodel"
    report: dict[str, Any] = {
        "package": export_and_check(checkpoint, package, args.library),
        "config": config,
        "library_sha256": digest(args.library),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": args.precision,
        "unexact_erf": args.unexact_erf,
    }
    eager, _ = load_checkpoint(checkpoint)
    if not isinstance(eager, GFISRV2):
        raise TypeError("The validation checkpoint must load as GFISRv2")
    eager.eval().cuda()
    eager_half = deepcopy(eager).half() if args.precision == "fp16" else None
    check_fp32_parity = args.precision == "fp32" and not args.unexact_erf
    options = {"unexact_erf": 1} if args.unexact_erf else {}
    plan_cache = args.output / "plans" / args.precision if args.precision != "fp32" else None
    settings = SessionOptions(
        precision=args.precision, dll_path=args.library, cache_dir=plan_cache, kernel_options=options
    )
    direct_settings = SessionOptions(
        precision=args.precision,
        dll_path=args.library,
        cache_dir=plan_cache,
        enable_cuda_graph=False,
        kernel_options=options,
    )
    with InferenceSession(package, settings) as graph, InferenceSession(package, direct_settings) as direct:
        report["stress"] = check_stress(graph, direct, eager, eager_half, sources, args.output, settings)
        report["dispatch"] = check_dispatch(graph, package, args.cli, args.output, settings)
    del eager, eager_half
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.pages:
        pages, setup = check_pages(
            package,
            settings,
            sources,
            args.reference_report.parent,
            args.output,
            check_fp32_parity,
            eager_half_reference,
            family,
        )
        report.update(pages=pages, setup=setup, summary=summarize(pages))
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
