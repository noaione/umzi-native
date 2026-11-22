"""Validate the packaged GateRV2 session against the untouched eager FP32 model.

The check exports the checkpoint twice, rejects malformed GateRV2 payloads, runs
the stress and odd-shape cases through CUDA Graph and direct execution, checks
device enqueue and cleanup, and measures both benchmark pages with batch-16
throughput and individual batch-1 tile latency.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import image_tensor
from benchmark_architectures import (
    FAMILIES,
    METRIC,
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
from resselt.archs.gaterv2.arch import GateRV2
from torch import Tensor

from umzi_native import GateRV2Model, InferenceSession, SessionOptions, peek_model
from umzi_native._model import GATERV2_ARCHITECTURE, GATERV2_FAMILY, GATERV3_FAMILY, HEADER_SIZE, SECTION
from umzi_native.exporter import export_model
from umzi_native.exporter._loader import load_checkpoint
from umzi_native.exporter._package import write_package

FAMILY = FAMILIES["gaterv2"]
NETWORK_TOLERANCE = {"atol": 1e-5, "rtol": 5e-5}
# Flat and periodic tiles drive the linear-attention denominator close to zero,
# which amplifies FP32 rounding in any implementation. When a case misses the
# eager tolerance, compare both FP32 paths with an FP64 run of the same model
# and require the session to stay within this factor of eager FP32's error.
FP64_ERROR_RATIO = 1.5


def file_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def sections(path: Path) -> dict[int, bytes]:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 20)[0]
    return {
        kind: data[offset : offset + length]
        for kind, _, offset, length in SECTION.iter_unpack(data[HEADER_SIZE : HEADER_SIZE + count * SECTION.size])
    }


def rejected(path: Path, options: SessionOptions, message: str) -> str:
    try:
        with InferenceSession(path, options):
            pass
    except RuntimeError as error:
        if message in str(error):
            return str(error)
        raise
    raise RuntimeError(f"{path.name} was accepted, expected {message!r}")


def check_rejections(model: Path, library: Path, root: Path) -> dict[str, str]:
    options = SessionOptions(dll_path=library)
    parts = sections(model)
    results: dict[str, str] = {}
    broken = root / "broken.ummodel"

    def package(family: int, schema: int, changed: dict[int, bytes]) -> Path:
        write_package(broken, family, schema, {**parts, **changed})
        return broken

    results["schema"] = rejected(package(GATERV2_FAMILY, 2, {}), options, "Unsupported GateRV2 model payload version")
    results["wrong_family"] = rejected(package(GATERV3_FAMILY, 1, {}), options, "GateRV3")
    architecture = list(GATERV2_ARCHITECTURE.unpack(parts[1]))
    architecture[3] = 8
    results["alignment"] = rejected(
        package(GATERV2_FAMILY, 1, {1: GATERV2_ARCHITECTURE.pack(*architecture)}), options, "architecture"
    )
    table = bytearray(parts[2])
    struct.pack_into("<Q", table, 8, 1)
    results["parameter_count"] = rejected(package(GATERV2_FAMILY, 1, {2: bytes(table)}), options, "parameter 0")
    results["truncated_table"] = rejected(package(GATERV2_FAMILY, 1, {2: parts[2][:-16]}), options, "parameter table")
    results["trailing_weights"] = rejected(
        package(GATERV2_FAMILY, 1, {3: parts[3] + bytes(256)}), options, "trailing GateRV2 weight"
    )
    for label, info in (("peek_schema", package(GATERV2_FAMILY, 2, {})),):
        problems = peek_model(info).problems
        if problems != ("Unsupported GateRV2 model payload version",):
            raise RuntimeError(f"Unexpected {label} problems: {problems}")
    results["precision"] = rejected(model, SessionOptions(dll_path=library, precision="fp16"), "FP32 only")
    results["kernel_option"] = rejected(
        model, SessionOptions(dll_path=library, kernel_options={"fuse_silu": 1}), "no kernel options"
    )
    with InferenceSession(model, options) as session:
        for shape, message in (((1, 3, 8, 9), "reflection padding"), ((1, 1, 32, 32), "RGB NCHW")):
            try:
                session.prepare(shape)
            except RuntimeError as error:
                if message not in str(error):
                    raise
                results[f"shape_{'x'.join(map(str, shape))}"] = str(error)
            else:
                raise RuntimeError(f"{shape} was accepted")
    broken.unlink(missing_ok=True)
    return results


def run_session(session: InferenceSession, sample: Tensor) -> Tensor:
    return torch.from_numpy(session.run(None, {"input": sample.numpy()})[0])


def within_tolerance(actual: Tensor, expected: Tensor) -> bool:
    return bool(torch.isclose(actual, expected, **NETWORK_TOLERANCE).all())


def fp64_check(reference64: GateRV2, sample: Tensor, expected: Tensor, actual: Tensor, label: str) -> dict[str, Any]:
    exact = reference64(sample.cuda().double()).cpu()
    eager_error = float((expected.double() - exact).abs().max())
    session_error = float((actual.double() - exact).abs().max())
    row = {
        "eager_fp32_max_abs_vs_fp64": eager_error,
        "session_max_abs_vs_fp64": session_error,
        "allowed_ratio": FP64_ERROR_RATIO,
        "eager_tolerance_violations": int((~torch.isclose(actual, expected, **NETWORK_TOLERANCE)).sum()),
    }
    if not session_error <= FP64_ERROR_RATIO * eager_error:
        raise AssertionError(f"{label}: session FP64 error {session_error:.3g} exceeds eager {eager_error:.3g}")
    return row


def check_stress(
    reference: GateRV2, sessions: dict[str, InferenceSession], cases: dict[str, Tensor], output: Path
) -> dict[str, Any]:
    reference64 = copy.deepcopy(reference).double()
    rows: dict[str, Any] = {}
    for label, sample in cases.items():
        expected = reference(sample.cuda()).cpu()
        row: dict[str, Any] = {"input_shape": list(sample.shape)}
        for mode, session in sessions.items():
            actual = run_session(session, sample)
            row[mode] = comparison(actual.numpy(), expected.numpy())
            if not row[mode]["finite"]:
                raise AssertionError(f"{mode} {label} produced non-finite output")
            if not within_tolerance(actual, expected):
                row[mode]["fp64"] = fp64_check(reference64, sample, expected, actual, f"{mode} {label}")
            if label == "checkerboard" and mode == "graph":
                row[mode]["ssimulacra2"] = score_stress_render(output / "stress-renders", label, expected, actual)
        if len(sample) > 1:
            graph = sessions["graph"]
            batched = run_session(graph, sample)
            row["vs_individual"] = [
                comparison(item.numpy(), run_session(graph, tile)[0].numpy())
                for item, tile in zip(batched, sample.split(1), strict=True)
            ]
        rows[label] = row
        note = " (FP64-anchored)" if "fp64" in row["graph"] else ""
        print(f"{label}: graph {row['graph'].get('psnr_db')} dB, max {row['graph']['max_abs']:.3g}{note}", flush=True)
    return rows


def check_enqueue(library: Path, model: Path, reference: GateRV2) -> dict[str, Any]:
    from umzi_native._abi import ShapeABI, check_status

    result: dict[str, Any] = {}
    with InferenceSession(model, SessionOptions(dll_path=library)) as session:
        dll, handle = session._dll, session._handle
        generator = torch.Generator().manual_seed(7)
        for shape in ((2, 3, 128, 128), (1, 3, 65, 79)):
            source = torch.rand(shape, generator=generator).cuda()
            target = torch.empty_like(source)
            stream = torch.cuda.Stream()
            dims = ShapeABI.from_dimensions(shape)
            nbytes = source.numel() * 4
            stream.wait_stream(torch.cuda.current_stream())
            check_status(
                dll,
                dll.umzi_session_enqueue(
                    handle, dims, source.data_ptr(), nbytes, target.data_ptr(), nbytes, stream.cuda_stream
                ),
            )
            stream.synchronize()
            expected = reference(source)
            if not within_tolerance(target, expected):
                raise AssertionError(f"Device enqueue differs from eager FP32 for {shape}")
            result["x".join(map(str, shape))] = comparison(target.cpu().numpy(), expected.cpu().numpy())
    return result


def check_cleanup(library: Path, model: Path) -> dict[str, Any]:
    torch.cuda.synchronize()
    free_before, _ = torch.cuda.mem_get_info()
    sample = np.random.default_rng(3).random((4, 3, 128, 128), dtype=np.float32)
    shapes = []
    for _ in range(3):
        with InferenceSession(model, SessionOptions(dll_path=library)) as session:
            for shape in ((4, 3, 128, 128), (1, 3, 65, 79), (2, 3, 128, 160), (4, 3, 128, 128)):
                session.run(None, {"input": np.ascontiguousarray(sample[: shape[0], :, : shape[2], : shape[3]])})
                shapes.append(session.get_session_info()["padded_shape"])
        gc.collect()
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    return {
        "padded_shapes": [list(shape) for shape in shapes[:4]],
        "device_free_before": free_before,
        "device_free_after": free_after,
        "device_free_delta": free_before - free_after,
    }


class SessionRunner(Runner):
    """Time a host session call; the session copies RGB through device memory."""

    def __init__(self, session: InferenceSession):
        super().__init__(lambda sample: sample, torch.float32)
        self.session = session

    def prepare(self, host: Tensor) -> Tensor:
        return host.contiguous()

    def measure(self, host: Tensor) -> tuple[Tensor, float]:
        sample = host.contiguous().numpy()
        self.session.prepare(sample.shape)
        torch.cuda.synchronize()
        started = time.perf_counter()
        (result,) = self.session.run(None, {"input": sample})
        return torch.from_numpy(result), time.perf_counter() - started


def check_pages(
    reference: GateRV2, library: Path, model: Path, sources: list[Path], output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not METRIC.is_file():
        raise FileNotFoundError(METRIC)
    options = SessionOptions(dll_path=library)
    pages = []
    with InferenceSession(model, options) as batched, InferenceSession(model, options) as single:
        batched.prepare((FAMILY.batch, 3, FAMILY.tile, FAMILY.tile))
        single.prepare((1, 3, FAMILY.tile, FAMILY.tile))
        runners: dict[int, Runner] = {1: SessionRunner(single), FAMILY.batch: SessionRunner(batched)}
        eager = Runner(reference, torch.float32)
        for path in sources:
            image = read_image(path)
            row, candidate = timed_page(image, path.name, runners, FAMILY, FAMILY.batch)
            reference_row, expected = timed_page(
                image, path.name, {1: eager, FAMILY.batch: eager}, FAMILY, FAMILY.batch
            )
            row.update(comparison(candidate.numpy(), expected.numpy()))
            row["within_eager_tolerance"] = within_tolerance(candidate, expected)
            if not row["finite"]:
                raise AssertionError(f"Non-finite output for {path.name}")
            ref_png, native_png = output / f"{path.stem}-fp32.png", output / f"{path.stem}-session.png"
            image_tensor(expected).save(ref_png)
            image_tensor(candidate).save(native_png)
            row["ssimulacra2"] = ssimulacra2(ref_png, native_png)
            row["differing_rgb_samples"] = int(
                np.count_nonzero(np.asarray(image_tensor(expected)) != np.asarray(image_tensor(candidate)))
            )
            row["reference_timing"] = reference_row
            pages.append(row)
            print(
                f"{path.name}: {row['psnr_db']} dB, SSIMULACRA2 {row['ssimulacra2']}, "
                f"batch-16 {row['batch_inference_seconds']:.3f} s",
                flush=True,
            )
    reference_pages = [page["reference_timing"] for page in pages]
    return pages, {"session": summarize(pages), "eager_fp32": summarize(reference_pages)}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True, help="umzi_native shared library under test")
    parser.add_argument("--output", type=Path, default=ROOT / "output/gaterv2-session")
    parser.add_argument("--pages", action="store_true", help="Also measure the two benchmark pages")
    args = parser.parse_args()
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    library = args.library.resolve()
    checkpoint = ROOT / "models" / FAMILY.checkpoint
    sources = sorted((ROOT / "samples" / FAMILY.samples).glob("*.jpg"))
    report: dict[str, Any] = {
        "checkpoint_sha256": digest(checkpoint),
        "library": str(library),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "output_tolerance": NETWORK_TOLERANCE,
        "inputs": {path.name: digest(path) for path in sources},
    }
    with tempfile.TemporaryDirectory(dir=args.output) as directory:
        root = Path(directory)
        first = export_model(checkpoint, root / "first.ummodel", kernel="gaterv2")
        second = export_model(checkpoint, root / "second.ummodel")
        if file_digest(first.path) != file_digest(second.path):
            raise RuntimeError("GateRV2 export is not deterministic")
        payload = first.payload
        if not isinstance(payload, GateRV2Model) or payload.checkpoint != report["checkpoint_sha256"]:
            raise RuntimeError("Exported GateRV2 metadata does not match the checkpoint")
        model = args.output / "gaterv2.ummodel"
        first.path.replace(model)
        report["package"] = {"sha256": file_digest(model), "size": model.stat().st_size, "deterministic": True}
        report["rejections"] = check_rejections(model, library, root)
    print("Export determinism and rejection checks passed", flush=True)

    reference, _ = load_checkpoint(checkpoint)
    if not isinstance(reference, GateRV2):
        raise TypeError("The validation checkpoint is not GateRV2")
    reference.eval().cuda()
    cases = dict(stress_inputs(FAMILY.tile, sources, FAMILY.batch))
    generator = torch.Generator().manual_seed(9)
    cases["rect-9x9"] = torch.rand((1, 3, 9, 9), generator=generator)
    cases["rect-31x47-batch3"] = torch.rand((3, 3, 31, 47), generator=generator)
    with (
        InferenceSession(model, SessionOptions(dll_path=library)) as graph,
        InferenceSession(model, SessionOptions(dll_path=library, enable_cuda_graph=False)) as direct,
    ):
        report["stress"] = check_stress(reference, {"graph": graph, "direct": direct}, cases, args.output)
        report["session_info"] = graph.get_session_info()
    report["enqueue"] = check_enqueue(library, model, reference)
    report["cleanup"] = check_cleanup(library, model)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.pages:
        report["pages"], report["summary"] = check_pages(reference, library, model, sources, args.output)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    sys.exit(main())
