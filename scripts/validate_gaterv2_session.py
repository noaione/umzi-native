"""Validate the packaged GateRV2 session against the untouched eager FP32 model.

The native session runs in mixed precision (FP32 RGB I/O and convolution
accumulation) or FP16. Both keep FP16 features and FP32 attention. The check
exports the checkpoint twice, rejects malformed GateRV2 payloads, runs the
stress and odd-shape cases through CUDA Graph and direct execution in both
precisions, checks batch consistency and cleanup, and measures both pages with
device-resident batch-16 throughput and individual batch-1 tile latency.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import struct
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
from umzi_native._abi import IoABI, ShapeABI, check_status
from umzi_native._model import GATERV2_ARCHITECTURE, GATERV2_FAMILY, GATERV3_FAMILY, HEADER_SIZE, SECTION
from umzi_native.exporter import export_model
from umzi_native.exporter._loader import load_checkpoint
from umzi_native.exporter._package import write_package

FAMILY = FAMILIES["gaterv2"]
PRECISIONS = {"mixed": torch.float32, "fp16": torch.float16}
# Direct execution, CUDA Graph replay, and batch-1 runs use separately timed
# cuBLASLt and cuDNN algorithms, so their roundings differ. Each must be as
# close to eager FP32 as the graph run: a coupled batch or a broken path would
# lose far more than this. FP16 flat tiles vary by about 1.3 dB between plans.
CONSISTENCY_MARGIN_DB = 3.0


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
    if peek_model(broken).problems != ("Unsupported GateRV2 model payload version",):
        raise RuntimeError(f"Unexpected reader problems: {peek_model(broken).problems}")
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
    results["unknown_option"] = rejected(
        model, SessionOptions(dll_path=library, kernel_options={"fuse_silu": 1}), "Unknown GateRV2 option"
    )
    results["tune_plans_value"] = rejected(
        model, SessionOptions(dll_path=library, kernel_options={"tune_plans": 2}), "tune_plans must be 0 or 1"
    )
    results["workspace_value"] = rejected(
        model, SessionOptions(dll_path=library, kernel_options={"workspace_mib": 2048}), "workspace_mib must be"
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


class DeviceSession:
    """Enqueue CUDA tensors on the current Torch stream through the session ABI."""

    def __init__(self, session: InferenceSession):
        self.session = session
        self.dtype = torch.float16 if session.get_inputs()[0].type == "tensor(float16)" else torch.float32

    def __call__(self, sample: Tensor) -> Tensor:
        if sample.dtype != self.dtype or not sample.is_cuda or not sample.is_contiguous():
            raise ValueError("The session expects a contiguous CUDA tensor of its I/O dtype")
        dims = ShapeABI.from_dimensions(sample.shape)
        io = IoABI(ctypes.sizeof(IoABI))
        dll, handle = self.session._dll, self.session._handle
        check_status(dll, dll.umzi_session_describe(handle, ctypes.byref(dims), ctypes.byref(io)))
        result = torch.empty(io.output.shape.dimensions_tuple(), dtype=self.dtype, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        check_status(
            dll,
            dll.umzi_session_enqueue(
                handle,
                ctypes.byref(dims),
                sample.data_ptr(),
                io.input.bytes,
                result.data_ptr(),
                io.output.bytes,
                stream,
            ),
        )
        return result


def consistent(candidate: Tensor, expected: Tensor, graph: dict[str, Any], message: str) -> dict[str, Any]:
    row = comparison(candidate.numpy(), expected.numpy())
    if not row["finite"]:
        raise AssertionError(f"{message} produced {row['nonfinite_values']} non-finite values")
    psnr, baseline = row["psnr_db"], graph["psnr_db"]
    if psnr is not None and baseline is not None and psnr < baseline - CONSISTENCY_MARGIN_DB:
        raise AssertionError(f"{message} ({psnr} dB against eager FP32; graph run {baseline} dB)")
    return row


def check_stress(
    reference: GateRV2, model: Path, library: Path, cases: dict[str, Tensor], output: Path
) -> dict[str, Any]:
    expected = {label: reference(sample.cuda()).cpu() for label, sample in cases.items()}
    rows: dict[str, Any] = {label: {"input_shape": list(sample.shape)} for label, sample in cases.items()}
    # Reuse the same algorithms across graph and direct dispatch. Independently
    # tuned FP16 plans can differ substantially on the one-token attention case.
    with tempfile.TemporaryDirectory(prefix="stress-plans-", dir=output) as directory:
        for precision, dtype in PRECISIONS.items():
            graph = DeviceSession(
                InferenceSession(model, SessionOptions(precision=precision, dll_path=library, cache_dir=directory))
            )
            direct = DeviceSession(
                InferenceSession(
                    model,
                    SessionOptions(precision=precision, enable_cuda_graph=False, dll_path=library, cache_dir=directory),
                )
            )
            try:
                for label, sample in cases.items():
                    x = sample.to(device="cuda", dtype=dtype)
                    actual = graph(x).float().cpu()
                    row = comparison(actual.numpy(), expected[label].numpy())
                    if not row["finite"]:
                        raise AssertionError(f"{precision} {label} produced non-finite output")
                    if not torch.equal(graph(x).float().cpu(), actual):
                        raise AssertionError(f"{precision} {label}: repeated CUDA Graph runs differ")
                    direct_output = direct(x).float().cpu()
                    if direct.session.get_session_info()["cache_status"] != "hit":
                        raise AssertionError(
                            f"{precision} {label}: direct session did not use the graph session's plans"
                        )
                    if not torch.equal(direct_output, actual):
                        raise AssertionError(f"{precision} {label}: direct and graph differ on matching plans")
                    if not torch.equal(direct(x).float().cpu(), direct_output):
                        raise AssertionError(f"{precision} {label}: repeated direct runs differ")
                    row["direct_vs_graph"] = comparison(direct_output.numpy(), actual.numpy())
                    row["direct"] = consistent(direct_output, expected[label], row, f"{precision} {label}: direct run")
                    if label in ("checkerboard", "noise"):
                        row["ssimulacra2"] = score_stress_render(
                            output / "stress-renders" / precision, label, expected[label], actual
                        )
                    if len(sample) > 1:
                        individual = torch.cat([graph(tile).float().cpu() for tile in x.split(1)])
                        row["vs_individual"] = comparison(actual.numpy(), individual.numpy())
                        row["individual"] = consistent(
                            individual, expected[label], row, f"{precision} {label}: batch-1 runs of the batch"
                        )
                    rows[label][precision] = row
                    note = f", SSIMULACRA2 {row['ssimulacra2']:.3f}" if "ssimulacra2" in row else ""
                    print(f"{precision} {label}: {row['psnr_db']:.3f} dB, max {row['max_abs']:.3g}{note}", flush=True)
                rows.setdefault("_session_info", {})[precision] = graph.session.get_session_info()
            finally:
                graph.session.close()
                direct.session.close()
    return rows


def check_host_io(reference: GateRV2, model: Path, library: Path) -> dict[str, Any]:
    """Host run must equal device enqueue for the same input."""
    sample = torch.rand((2, 3, 65, 79), generator=torch.Generator().manual_seed(11))
    result = {}
    for precision, dtype in PRECISIONS.items():
        with InferenceSession(model, SessionOptions(precision=precision, dll_path=library)) as session:
            x = sample.to(dtype)
            host = torch.from_numpy(session.run(None, {"input": x.numpy()})[0])
            device = DeviceSession(session)(x.cuda()).cpu()
            if not torch.equal(host, device):
                raise AssertionError(f"{precision}: host run differs from device enqueue in the same session")
            result[precision] = comparison(host.float().numpy(), reference(sample.cuda()).cpu().numpy())
    return result


def check_options(reference: GateRV2, model: Path, library: Path) -> dict[str, Any]:
    """Each kernel option keeps the output finite and close to eager FP32."""
    sample = torch.rand((2, 3, 65, 79), generator=torch.Generator().manual_seed(13))
    expected = reference(sample.cuda()).cpu()
    result = {}
    for label, options in (
        ("default", {}),
        ("heuristic_plans", {"tune_plans": 0}),
        ("no_workspace", {"workspace_mib": 0}),
    ):
        for precision, dtype in PRECISIONS.items():
            with InferenceSession(
                model, SessionOptions(precision=precision, kernel_options=options, dll_path=library)
            ) as session:
                actual = DeviceSession(session)(sample.to(device="cuda", dtype=dtype)).float().cpu()
                info = session.get_session_info()
            row = comparison(actual.numpy(), expected.numpy())
            row["diagnostics"] = info["diagnostics"]
            row["buffer_bytes"] = info["buffer_bytes"]
            psnr = row.get("psnr_db")
            if not row["finite"] or (psnr is not None and psnr < 60):
                raise AssertionError(f"{precision} with {options}: {row.get('psnr_db')} dB")
            if "tune_plans" in options and info["diagnostics"]["timed_plans"]:
                raise AssertionError(f"{precision}: the heuristic-only session timed plans")
            if "workspace_mib" in options and info["buffer_bytes"]["workspace"]:
                raise AssertionError(f"{precision}: the zero-workspace session allocated workspace")
            result[f"{label}-{precision}"] = row
    return result


def check_cleanup(library: Path, model: Path) -> dict[str, Any]:
    torch.cuda.synchronize()
    free_before, _ = torch.cuda.mem_get_info()
    shapes = []
    for precision, dtype in (*PRECISIONS.items(), ("mixed", torch.float32)):
        sample = np.random.default_rng(3).random((4, 3, 128, 160)).astype(torch.empty(0, dtype=dtype).numpy().dtype)
        with InferenceSession(model, SessionOptions(precision=precision, dll_path=library)) as session:
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


def check_plan_cache(library: Path, model: Path, output: Path) -> dict[str, Any]:
    """Reopen tuned plans, verify key isolation, and recover from cache failures."""
    generator = torch.Generator().manual_seed(702)
    samples = {
        batch: torch.rand((batch, 3, 128, 128), generator=generator, dtype=torch.float32).cuda()
        for batch in (1, FAMILY.batch)
    }

    def run(
        precision: str, batch: int, cache: Path, expected_status: str, **options: int
    ) -> tuple[Tensor, dict[str, Any]]:
        sample = samples[batch].to(PRECISIONS[precision])
        with InferenceSession(
            model, SessionOptions(precision=precision, dll_path=library, cache_dir=cache, kernel_options=options)
        ) as session:
            torch.cuda.synchronize()
            started = time.perf_counter()
            session.prepare(sample.shape)
            prepare_ms = (time.perf_counter() - started) * 1000
            prediction = DeviceSession(session)(sample).cpu()
            info = session.get_session_info()
        if info["cache_status"] != expected_status:
            raise AssertionError(
                f"{precision} batch {batch}: expected cache {expected_status}, got "
                f"{info['cache_status']} ({info['cache_message']})"
            )
        if expected_status == "hit" and info["diagnostics"]["timed_plans"]:
            raise AssertionError(f"{precision} batch {batch}: retimed a cache hit")
        if not torch.isfinite(prediction).all():
            raise AssertionError(f"{precision} batch {batch}: cache validation returned non-finite values")
        return prediction, {"prepare_ms": prepare_ms, **info}

    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="cache-check-", dir=output) as temporary:
        cache = Path(temporary)
        for precision in PRECISIONS:
            for batch in samples:
                first, miss = run(precision, batch, cache, "miss")
                reopened, hit = run(precision, batch, cache, "hit")
                if not torch.equal(first, reopened):
                    raise AssertionError(f"{precision} batch {batch}: plan cache changed the output")
                results[f"{precision}-b{batch}"] = {"miss": miss, "hit": hit}
        if len(list(cache.glob("gaterv2_*.umplan"))) != 4:
            raise AssertionError("GateRV2 cache keys did not separate batch sizes and precisions")

        _, heuristic = run("mixed", 1, cache, "miss", tune_plans=0)
        first, zero_workspace = run("mixed", 1, cache, "miss", workspace_mib=0)
        reopened, zero_hit = run("mixed", 1, cache, "hit", workspace_mib=0)
        if not torch.equal(first, reopened) or zero_hit["buffer_bytes"]["workspace"]:
            raise AssertionError("The zero-workspace plan cache changed the session")
        if len(list(cache.glob("gaterv2_*.umplan"))) != 6:
            raise AssertionError("GateRV2 cache keys did not separate tuning and workspace options")
        results["option_isolation"] = {"heuristic": heuristic, "zero_workspace": zero_workspace, "hit": zero_hit}

        corrupt_cache = cache / "corrupt"
        run("mixed", 1, corrupt_cache, "miss")
        plan = next(corrupt_cache.glob("gaterv2_*.umplan"))
        data = bytearray(plan.read_bytes())
        data[-1] ^= 1
        plan.write_bytes(data)
        repaired, rebuilt = run("mixed", 1, corrupt_cache, "rebuilt")
        reopened, hit = run("mixed", 1, corrupt_cache, "hit")
        if not torch.equal(repaired, reopened):
            raise AssertionError("The rebuilt GateRV2 cache changed the output")
        unavailable = cache / "regular-file"
        unavailable.touch()
        _, failed = run("mixed", 1, unavailable, "unavailable")
        results["recovery"] = {"rebuilt": rebuilt, "hit": hit, "unavailable": failed}
    return results


def check_pages(reference: GateRV2, library: Path, model: Path, sources: list[Path], output: Path) -> dict[str, Any]:
    if not METRIC.is_file():
        raise FileNotFoundError(METRIC)
    eager = Runner(reference, torch.float32)
    expected_pages = {}
    reference_rows = []
    for path in sources:
        row, expected = timed_page(read_image(path), path.name, {1: eager, FAMILY.batch: eager}, FAMILY, FAMILY.batch)
        expected_pages[path.name] = expected
        image_tensor(expected).save(output / f"{path.stem}-fp32.png")
        reference_rows.append(row)
    result: dict[str, Any] = {"eager_fp32": {"pages": reference_rows, "summary": summarize(reference_rows)}}
    for precision, dtype in PRECISIONS.items():
        options = SessionOptions(precision=precision, dll_path=library)
        # Batch-1 latency uses a separate session prepared once for batch 1.
        with InferenceSession(model, options) as batched, InferenceSession(model, options) as single:
            batched.prepare((FAMILY.batch, 3, FAMILY.tile, FAMILY.tile))
            single.prepare((1, 3, FAMILY.tile, FAMILY.tile))
            runners = {1: Runner(DeviceSession(single), dtype), FAMILY.batch: Runner(DeviceSession(batched), dtype)}
            pages = []
            for path in sources:
                row, candidate = timed_page(read_image(path), path.name, runners, FAMILY, FAMILY.batch)
                expected = expected_pages[path.name]
                row.update(comparison(candidate.numpy(), expected.numpy()))
                if not row["finite"]:
                    raise AssertionError(f"{precision}: non-finite output for {path.name}")
                rendered = output / f"{path.stem}-{precision}.png"
                image_tensor(candidate).save(rendered)
                row["ssimulacra2"] = ssimulacra2(output / f"{path.stem}-fp32.png", rendered)
                pages.append(row)
                print(
                    f"{precision} {path.name}: {row['psnr_db']:.3f} dB, SSIMULACRA2 {row['ssimulacra2']:.3f}, "
                    f"batch-16 {row['batch_inference_seconds']:.3f} s",
                    flush=True,
                )
            result[precision] = {"pages": pages, "summary": summarize(pages)}
    return result


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
    report["stress"] = check_stress(reference, model, library, cases, args.output)
    report["host_io"] = check_host_io(reference, model, library)
    report["options"] = check_options(reference, model, library)
    report["plan_cache"] = check_plan_cache(library, model, args.output)
    report["cleanup"] = check_cleanup(library, model)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.pages:
        report["pages"] = check_pages(reference, library, model, sources, args.output)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
