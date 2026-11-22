"""Compare packaged GateRV2 with TensorRT-RTX or a control DLL in paired page runs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from benchmark import image_tensor
from benchmark_architectures import (
    ROOT,
    Runner,
    comparison,
    digest,
    open_rtx_runner,
    read_image,
    ssimulacra2,
    summarize,
    timed_page,
)
from torch import Tensor
from validate_gaterv2_session import FAMILY, PRECISIONS, DeviceSession

from umzi_native import GateRV2Model, InferenceSession, SessionOptions, peek_model

BACKENDS = ("native-mixed", "native-fp16", "rtx")
BASELINE = ROOT / "output/architecture-baselines/gaterv2/batch-16"
TIMING_KEYS = ("batch_inference_seconds", "batch_input_mpx_per_second", "average_batch1_tile_latency_ms")


def read_report(path: Path, config: dict[str, Any], backend: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "complete" or report["backend"] != backend:
        raise ValueError(f"Expected a completed {backend} report at {path}")
    if any(report["config"][key] != value for key, value in config.items()):
        raise ValueError(f"Checkpoint, images or tiling differ in {path}")
    return report


def measure_pages(
    sources: list[Path],
    runners: dict[str, dict[int, Runner]],
    references: Path,
    output: Path,
    rounds: int,
) -> dict[str, Any]:
    backends = tuple(runners)
    history: dict[str, list[dict[str, Any]]] = {name: [] for name in backends}
    first_results: dict[tuple[str, str], Tensor] = {}
    images = {path.name: read_image(path) for path in sources}
    for round_index in range(rounds):
        start = round_index % len(backends)
        order = backends[start:] + backends[:start]
        print(f"Round {round_index + 1}: {', '.join(order)}", flush=True)
        pages: dict[str, list[dict[str, Any]]] = {name: [] for name in backends}
        for label, image in images.items():
            for name in order:
                row, candidate = timed_page(image, label, runners[name], FAMILY, FAMILY.batch)
                pages[name].append(row)
                if round_index == 0:
                    first_results[name, label] = candidate
                elif not torch.equal(candidate, first_results[name, label]):
                    raise RuntimeError(f"{name} changed output on {label} in round {round_index + 1}")
        for name in order:
            history[name].append({"round": round_index + 1, "pages": pages[name], "summary": summarize(pages[name])})

    report: dict[str, Any] = {}
    for name in backends:
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        quality: list[dict[str, Any]] = []
        for label in images:
            stem = Path(label).stem
            reference = np.load(references / f"{stem}.npy", mmap_mode="r")
            candidate = first_results[name, label]
            row = {"image": label, **comparison(candidate.numpy(), reference)}
            if not row["finite"]:
                raise RuntimeError(f"{name} produced non-finite output for {label}")
            image_tensor(candidate).save(target / f"{stem}.png")
            row["ssimulacra2"] = ssimulacra2(references / f"{stem}.png", target / f"{stem}.png")
            if name.startswith("native-"):
                control = first_results.get((name.replace("native-", "control-", 1), label))
                if control is not None:
                    row["bitwise_vs_control"] = torch.equal(candidate, control)
            quality.append(row)
        total_mse = sum(row["mse"] * row["values"] for row in quality) / sum(row["values"] for row in quality)
        trials = history[name]
        report[name] = {
            "rounds": trials,
            "median": {key: statistics.median(trial["summary"][key] for trial in trials) for key in TIMING_KEYS},
            "median_per_image": {
                label: {key: statistics.median(trial["pages"][index][key] for trial in trials) for key in TIMING_KEYS}
                for index, label in enumerate(images)
            },
            "quality": {
                "pages": quality,
                "psnr_db": None if total_mse == 0 else -10 * float(np.log10(total_mse)),
                "ssimulacra2_mean": statistics.mean(row["ssimulacra2"] for row in quality),
                "ssimulacra2_min": min(row["ssimulacra2"] for row in quality),
            },
        }
    return report


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--control-library", type=Path, help="Compare two native DLLs using heuristic plans")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, default=BASELINE / "fp32/report.json")
    parser.add_argument("--rtx-report", type=Path, default=BASELINE / "rtx/report.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, help="Run a whole number of backend-order rotations")
    args = parser.parse_args()
    libraries = {"native": args.library}
    backends = BACKENDS
    kernel_options: dict[str, int] = {}
    if args.control_library is not None:
        libraries = {"control": args.control_library, **libraries}
        backends = tuple(f"{build}-{precision}" for precision in PRECISIONS for build in libraries)
        kernel_options["tune_plans"] = 0
    rounds = len(backends) if args.rounds is None else args.rounds
    if rounds < len(backends) or rounds % len(backends):
        parser.error(f"--rounds must be a positive multiple of {len(backends)}")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    payload = peek_model(args.model).payload
    if not isinstance(payload, GateRV2Model):
        raise ValueError("The benchmark requires a packaged GateRV2 model")
    sources = sorted((ROOT / "samples" / FAMILY.samples).glob("*.jpg"))
    config = {
        "checkpoint_sha256": payload.checkpoint,
        "images": {path.name: digest(path) for path in sources},
        "tile": FAMILY.tile,
        "batch": FAMILY.batch,
        "overlap": FAMILY.overlap,
        "scale": FAMILY.scale,
    }
    read_report(args.reference_report, config, "fp32")
    rtx = read_report(args.rtx_report, config, "rtx") if "rtx" in backends else None
    args.output.mkdir(parents=True, exist_ok=True)
    setup: dict[str, Any] = {name: {} for name in backends}
    runners: dict[str, dict[int, Runner]] = {name: {} for name in backends}
    with ExitStack() as stack:
        for batch in (1, FAMILY.batch):
            shape = (batch, 3, FAMILY.tile, FAMILY.tile)
            if rtx is not None:
                graph_info = rtx["setup"][str(batch)]
                graph_path = args.rtx_report.parent / graph_info["graph"]
                if digest(graph_path) != graph_info["graph_sha256"]:
                    raise ValueError(f"Graph hash differs from the RTX report for {graph_path}")
                graph = onnx.load(graph_path, load_external_data=False)
                rtx_setup = setup["rtx"][str(batch)] = {"graph_sha256": graph_info["graph_sha256"]}
                runners["rtx"][batch] = open_rtx_runner(
                    graph_path, graph.graph.input[0].name, batch, FAMILY, rtx_setup, output_dir=args.output
                )
            for build, library in libraries.items():
                for precision, dtype in PRECISIONS.items():
                    name = f"{build}-{precision}"
                    options = SessionOptions(precision=precision, dll_path=library, kernel_options=kernel_options)
                    session = stack.enter_context(InferenceSession(args.model, options))
                    session.prepare(shape)
                    runner = Runner(DeviceSession(session), dtype)
                    runner.measure(torch.zeros(shape))
                    runners[name][batch] = runner
                    setup[name][str(batch)] = session.get_session_info()
        results = measure_pages(sources, runners, args.reference_report.parent, args.output, rounds)
    report = {
        "config": config,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "library_sha256": digest(args.library),
        "package_sha256": digest(args.model),
        "setup": setup,
        "results": results,
    }
    if args.control_library is not None:
        report["control_library_sha256"] = digest(args.control_library)
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
