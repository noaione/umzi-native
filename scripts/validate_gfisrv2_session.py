"""Validate the packaged GFISRv2 FP32 session against untouched eager FP32."""

from __future__ import annotations

import argparse
import json
import os
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
    graph: InferenceSession, direct: InferenceSession, eager: GFISRV2, sources: list[Path], output: Path, unexact: bool
) -> dict[str, Any]:
    runner, uncaptured = DeviceSession(graph), DeviceSession(direct)
    cases = dict(stress_inputs(FAMILY.tile, sources, FAMILY.batch))
    for height, width in ((1, 1), (1, 7), (2, 2), (3, 5), (8, 9)):
        cases[f"tiny-{height}x{width}"] = torch.full((1, 3, height, width), 0.5)
    results = {}
    for name, host in cases.items():
        source = host.cuda()
        expected, actual = eager(source).cpu(), runner(source).cpu()
        row = raw_metrics(actual.numpy(), expected.numpy()) if unexact else compare_stage(name, actual, expected)
        if not row["finite"]:
            raise AssertionError(f"Non-finite GFISRv2 output for {name}")
        if not torch.equal(actual, uncaptured(source).cpu()):
            raise AssertionError(f"GFISRv2 graph and direct output differ for {name}")
        row.update(input_shape=list(source.shape), output_shape=list(actual.shape), graph_direct_match=True)
        if name in ("gray", "gray-128", "checkerboard", "noise"):
            row["ssimulacra2"] = score_stress_render(output / "stress-renders", name, expected, actual)
        if name in ("gray", "gray-128"):
            directory = output / "stress-raw" / name
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "input.npy", host.numpy())
            np.save(directory / "eager-fp32.npy", expected.numpy())
            np.save(directory / "native-fp32.npy", actual.numpy())
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
                raw_metrics(item.numpy(), runner(tile).cpu().numpy())
                if unexact
                else compare_stage(f"{name} tile {i}", item, runner(tile).cpu())
                for i, (item, tile) in enumerate(zip(actual.split(1), source.split(1), strict=True))
            ]
        results[name] = row
        print(f"{name}: max absolute error {row['max_abs']:.6g}", flush=True)
    return results


def check_dispatch(
    graph: InferenceSession, model: Path, cli: Path | None, output: Path, unexact: bool
) -> dict[str, bool]:
    generator = torch.Generator().manual_seed(717)
    sample = torch.rand((2, 3, 31, 45), generator=generator)
    (host,) = graph.run(None, {"input": sample.numpy()})
    source = sample.cuda()
    alternate = torch.cuda.Stream()
    alternate.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(alternate):
        device = DeviceSession(graph)(source)
    alternate.synchronize()
    if host.shape != (2, 3, 124, 180) or not np.array_equal(host, device.cpu().numpy()):
        raise AssertionError("GFISRv2 host and alternate CUDA stream outputs differ")
    if graph.get_session_info()["diagnostics"]["unexact_erf"] != int(unexact):
        raise AssertionError("GFISRv2 reported a different erf mode")
    results = {"host_device_alternate_stream_match": True}
    if cli is not None:
        with TemporaryDirectory(dir=output) as directory:
            temporary = Path(directory)
            switches = ("--unexact-erf",) if unexact else ()
            reference = cli_output(
                cli, model, "fp32", temporary / "plans", sample.numpy(), temporary, host.shape, switches
            )
        if not np.array_equal(host, reference):
            raise AssertionError("GFISRv2 CLI and Python outputs differ")
        results["cli_host_match"] = True
    return results


def check_pages(
    package: Path, settings: SessionOptions, sources: list[Path], reference: Path, output: Path, unexact: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, setup = [], {}
    with ExitStack() as stack:
        runners = {}
        for batch in (1, FAMILY.batch):
            session = stack.enter_context(InferenceSession(package, settings))
            shape = (batch, 3, FAMILY.tile, FAMILY.tile)
            session.prepare(shape)
            runner = Runner(DeviceSession(session), torch.float32)
            runner.measure(torch.zeros(shape))
            runners[batch] = runner
            setup[str(batch)] = session.get_session_info()
        for path in sources:
            row, actual = timed_page(read_image(path), path.name, runners, FAMILY, FAMILY.batch)
            expected = np.load(reference / f"{path.stem}.npy", mmap_mode="r")
            if not unexact:
                np.testing.assert_allclose(actual.numpy(), expected, err_msg=path.name, **CUFFT_TOLERANCE)
            row.update(raw_metrics(actual.numpy(), expected))
            if not row["finite"]:
                raise AssertionError(f"Non-finite GFISRv2 output for {path.name}")
            rendered = output / f"{path.stem}-session-fp32.png"
            image_tensor(actual).save(rendered)
            row["ssimulacra2"] = ssimulacra2(reference / path.name, rendered)
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
    parser.add_argument(
        "--unexact-erf", action="store_true", help="Measure CUDA erff without asserting reference parity"
    )
    args = parser.parse_args()
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
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
    if args.pages:
        ref = json.loads(args.reference_report.read_text(encoding="utf-8"))
        if (
            ref["backend"] != "fp32"
            or ref["status"] != "complete"
            or any(ref["config"][key] != value for key, value in config.items())
        ):
            raise ValueError("The saved FP32 report has a different checkpoint, sample or tiling configuration")
    if args.output is None:
        args.output = ROOT / "output/gfisrv2-session" / ("unexact-erf" if args.unexact_erf else "validated")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    package = args.output / "gfisrv2.ummodel"
    report: dict[str, Any] = {
        "package": export_and_check(checkpoint, package, args.library),
        "config": config,
        "library_sha256": digest(args.library),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "unexact_erf": args.unexact_erf,
    }
    eager, _ = load_checkpoint(checkpoint)
    if not isinstance(eager, GFISRV2):
        raise TypeError("The validation checkpoint must load as GFISRv2")
    eager.eval().cuda()
    options = {"unexact_erf": 1} if args.unexact_erf else {}
    settings = SessionOptions(precision="fp32", dll_path=args.library, kernel_options=options)
    direct_settings = SessionOptions(
        precision="fp32", dll_path=args.library, enable_cuda_graph=False, kernel_options=options
    )
    with InferenceSession(package, settings) as graph, InferenceSession(package, direct_settings) as direct:
        report["stress"] = check_stress(graph, direct, eager, sources, args.output, args.unexact_erf)
        report["dispatch"] = check_dispatch(graph, package, args.cli, args.output, args.unexact_erf)
    del eager
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.pages:
        pages, setup = check_pages(
            package, settings, sources, args.reference_report.parent, args.output, args.unexact_erf
        )
        report.update(pages=pages, setup=setup, summary=summarize(pages))
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
