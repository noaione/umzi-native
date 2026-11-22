"""Compare staged GFISRv2 FP32 Fourier operations with the untouched checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import image_tensor
from benchmark_architectures import (
    FAMILIES,
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
from reference.cufft import Cufft
from reference.gfisrv2_native import NativeGFISRV2, TorchTransforms, fourier_unit
from reference.native_stages import Trace
from resselt.archs.gfisrv2.arch import GFISRV2, FourierUnit
from torch import Tensor
from torch.nn import functional
from torch.overrides import TorchFunctionMode

from umzi_native.exporter._loader import load_checkpoint

FAMILY = FAMILIES["gfisrv2"]
BASELINE = ROOT / "output/architecture-baselines/gfisrv2/batch-2/fp32"
CUFFT_TOLERANCE = {"atol": 2e-5, "rtol": 5e-5}


class FourierTrace(TorchFunctionMode):
    """Observe the functional calls made by the unmodified Fourier unit."""

    def __init__(self, record: Trace):
        self.record = record

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        parameters = kwargs or {}
        if func is torch.fft.irfft2:
            self.record("inverse_spectrum", args[0] if args else parameters["input"])
        result = func(*args, **parameters)
        if func is torch.fft.rfft2:
            self.record("forward_spectrum", result)
        elif func is functional.gelu:
            self.record("gelu", result)
        elif func is torch.fft.irfft2:
            self.record("inverse", result)
        return result


def reference_fourier(unit: FourierUnit, source: Tensor) -> dict[str, Tensor]:
    stages: dict[str, Tensor] = {}

    def record(name: str, value: Tensor) -> None:
        stages[name] = value.cpu()

    handles = [
        unit.rn.register_forward_pre_hook(lambda _, args: record("packed", args[0])),
        unit.rn.register_forward_hook(lambda _, _args, value: record("spectral_norm", value)),
        unit.fpe.register_forward_hook(lambda _, _args, value: record("position", value)),
        unit.fdc.register_forward_pre_hook(lambda _, args: record("position_residual", args[0])),
        unit.fdc.register_forward_hook(lambda _, _args, value: record("projected", value)),
    ]
    try:
        with FourierTrace(record):
            result = unit(source)
        record("output", result)
    finally:
        for handle in handles:
            handle.remove()
    return stages


def raw_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    row = comparison(actual, expected)
    row["bitwise_match"] = np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    return row


def compare_stage(label: str, actual: Tensor, expected: Tensor, *, exact: bool = False) -> dict[str, Any]:
    torch.testing.assert_close(
        actual,
        expected,
        atol=0 if exact else CUFFT_TOLERANCE["atol"],
        rtol=0 if exact else CUFFT_TOLERANCE["rtol"],
        msg=lambda detail: f"{label}\n{detail}",
    )
    if actual.is_complex():
        actual, expected = torch.view_as_real(actual), torch.view_as_real(expected)
    return raw_metrics(actual.numpy(), expected.numpy())


def compare_fourier(name: str, unit: FourierUnit, source: Tensor, cufft: Cufft) -> dict[str, Any]:
    expected = reference_fourier(unit, source)
    results: dict[str, Any] = {}
    for backend, transforms in (("torch", TorchTransforms()), ("cufft", cufft)):
        actual: dict[str, Tensor] = {}
        fourier_unit(
            unit, source, transforms, lambda stage, value, actual=actual: actual.__setitem__(stage, value.cpu())
        )
        if tuple(actual) != tuple(expected):
            raise AssertionError(f"{name}: trace coverage differs, expected {tuple(expected)}, saw {tuple(actual)}")
        stages = {
            stage: compare_stage(f"{name}/{backend}/{stage}", actual[stage], reference, exact=backend == "torch")
            for stage, reference in expected.items()
        }
        results[backend] = {
            "first_difference": next((stage for stage, row in stages.items() if row["max_abs"]), None),
            "stages": stages,
        }
    return results


def check_fourier(reference: GFISRV2, cufft: Cufft) -> dict[str, Any]:
    units = [(name, module) for name, module in reference.named_modules() if isinstance(module, FourierUnit)]
    if len(units) != 24:
        raise ValueError(f"Expected 24 Fourier units, found {len(units)}")
    generator = torch.Generator(device="cuda").manual_seed(496)
    shape = (1, 30, 16, 20)
    odd = (2, 30, 17, 23)
    ramp = torch.arange(int(np.prod(shape)), device="cuda", dtype=torch.float32).reshape(shape)
    impulse = torch.zeros(odd, device="cuda")
    impulse[0, 0, 0, 2] = 1
    impulse[1, 23, -2, -1] = -0.5
    y, x = torch.arange(odd[-2], device="cuda"), torch.arange(odd[-1], device="cuda")
    checker = ((y[:, None] + x) % 2).float().expand(odd).contiguous()
    cases = {
        "ramp": ramp / ramp.numel(),
        "black": torch.zeros(shape, device="cuda"),
        "gray": torch.full(shape, 0.5, device="cuda"),
        "odd-impulses": impulse,
        "odd-checkerboard": checker,
        "odd-distinct-batch": torch.randn(odd, device="cuda", generator=generator),
        "tiny-width": torch.randn((1, 30, 7, 1), device="cuda", generator=generator),
        "tile": torch.randn((1, 30, FAMILY.tile, FAMILY.tile), device="cuda", generator=generator),
    }
    rows: dict[str, Any] = {}
    for case, sample in cases.items():
        tested = units[:1] if case == "tile" else units[:4]
        rows[case] = {
            "shape": list(sample.shape),
            "units": {name: compare_fourier(f"{case}/{name}", unit, sample, cufft) for name, unit in tested},
        }
        print(f"Fourier {case}: {len(tested)} units matched", flush=True)
    return rows


def reference_network_trace(reference: GFISRV2, sample: Tensor) -> dict[str, Tensor]:
    stages: dict[str, Tensor] = {}

    def hook(name: str):
        def record(_module, _args, value):
            stages[name] = value.cpu()

        return record

    handles = [reference.in_to_dim.register_forward_hook(hook("in_to_dim"))]
    handles += [
        block.register_forward_hook(hook(f"gfisr_body.{i}")) for i, block in enumerate(reference.gfisr_body[:-3])
    ]
    handles += [
        reference.gfisr_body.register_forward_hook(hook("tail")),
        reference.upscale.register_forward_pre_hook(
            lambda _, args: stages.__setitem__("feature_residual", args[0].cpu())
        ),
    ]
    try:
        stages["output"] = reference(sample).cpu()
    finally:
        for handle in handles:
            handle.remove()
    return stages


def check_network_trace(candidate: NativeGFISRV2, reference: GFISRV2) -> dict[str, Any]:
    def trace_one(sample: Tensor) -> dict[str, Any]:
        expected = reference_network_trace(reference, sample)
        stages: dict[str, Any] = {}

        def compare(name: str, value: Tensor) -> None:
            stages[name] = compare_stage(name, value.cpu(), expected.pop(name))

        candidate(sample, compare)
        if expected:
            raise AssertionError(f"Missing network stages: {tuple(expected)}")
        return stages

    generator = torch.Generator(device="cuda").manual_seed(130)
    rows = {}
    for shape in ((1, 3, 32, 40), (2, 3, 31, 45)):
        sample = torch.rand(shape, device="cuda", generator=generator)
        rows["x".join(map(str, shape))] = trace_one(sample)
        print(f"Network {shape}: {len(rows['x'.join(map(str, shape))])} stages matched", flush=True)
    return rows


def check_stress(candidate: NativeGFISRV2, reference: GFISRV2, sources: list[Path], output: Path) -> dict[str, Any]:
    cases = dict(stress_inputs(FAMILY.tile, sources, FAMILY.batch))
    for height, width in ((1, 1), (1, 7), (2, 2), (3, 5), (8, 9)):
        cases[f"tiny-{height}x{width}"] = torch.full((1, 3, height, width), 0.5)
    rows = {}
    for name, host in cases.items():
        source = host.cuda()
        expected, actual = reference(source).cpu(), candidate(source).cpu()
        row = compare_stage(name, actual, expected)
        row["input_shape"] = list(source.shape)
        row["output_shape"] = list(actual.shape)
        if name in ("gray", "gray-128", "checkerboard", "noise"):
            row["ssimulacra2"] = score_stress_render(output / "stress-renders", name, expected, actual)
        if source.size(0) > 1:
            row["vs_individual"] = [
                compare_stage(f"{name} tile {index}", item, candidate(tile).cpu())
                for index, (item, tile) in enumerate(zip(actual.split(1), source.split(1), strict=True))
            ]
        rows[name] = row
        print(f"{name}: max absolute error {row['max_abs']:.6g}", flush=True)
    return rows


def check_pages(
    model: GFISRV2, library: Path | None, sources: list[Path], baseline: Path, output: Path
) -> list[dict[str, Any]]:
    rows = []
    with ExitStack() as stack:
        runners = {}
        for batch in (1, FAMILY.batch):
            # Each batch retains its own warmed plans during both page passes.
            transforms = stack.enter_context(Cufft(library))
            runner = Runner(NativeGFISRV2(model, transforms), torch.float32)
            runner.measure(torch.zeros((batch, 3, FAMILY.tile, FAMILY.tile)))
            runners[batch] = runner
        for path in sources:
            row, actual = timed_page(read_image(path), path.name, runners, FAMILY, FAMILY.batch)
            reference = np.load(baseline / f"{path.stem}.npy", mmap_mode="r")
            np.testing.assert_allclose(actual.numpy(), reference, err_msg=path.name, **CUFFT_TOLERANCE)
            row.update(raw_metrics(actual.numpy(), reference))
            rendered = output / f"{path.stem}-cufft-fp32.png"
            image_tensor(actual).save(rendered)
            row["ssimulacra2"] = ssimulacra2(baseline / path.name, rendered)
            rows.append(row)
            print(f"{path.name}: max error {row['max_abs']:.6g}, SSIMULACRA2 {row['ssimulacra2']}", flush=True)
    return rows


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cufft-library", type=Path, help="Override the path to the cuFFT shared library")
    parser.add_argument("--output", type=Path, default=ROOT / "output/gfisrv2-native-validation")
    parser.add_argument("--reference-report", type=Path, default=BASELINE / "report.json")
    parser.add_argument("--pages", action="store_true", help="Also validate and time the three MangaScale pages")
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
    reference, _ = load_checkpoint(checkpoint)
    if not isinstance(reference, GFISRV2):
        raise TypeError("The validation checkpoint must load as GFISRv2")
    reference.eval().cuda()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "report.json"
    with Cufft(args.cufft_library) as cufft:
        candidate = NativeGFISRV2(reference, cufft)
        report: dict[str, Any] = {
            "config": config,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "resselt": version("resselt"),
            "cufft": cufft.version(),
            "cufft_tolerance": CUFFT_TOLERANCE,
            "fourier": check_fourier(reference, cufft),
            "network_traces": check_network_trace(candidate, reference),
            "stress": check_stress(candidate, reference, sources, args.output),
        }
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.pages:
        report["pages"] = check_pages(reference, args.cufft_library, sources, args.reference_report.parent, args.output)
        report["summary"] = summarize(report["pages"])
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
