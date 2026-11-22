"""Check an installed umzi_native package against the native GateRV3 CLI.

The check needs only NumPy and the package under test. It records which
cuBLASLt and cuDNN files the process loaded, so a run shows whether they came
from CUDA PyTorch or from umzi_native.load_cuda_libraries().
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

CASES = ((1, 3, 128, 128), (3, 3, 65, 79))
WINDOWS = sys.platform == "win32"
CUDA_NAMES = (
    ("cublasLt64_13.dll", "cudnn64_9.dll", "cudnn_graph64_9.dll", "cufft64_12.dll")
    if WINDOWS
    else ("libcublasLt.so.13", "libcudnn.so.9", "libcudnn_graph.so.9", "libcufft.so.12")
)


def loaded_cuda_libraries() -> dict[str, str]:
    """Return the file each CUDA dependency was loaded from in this process."""
    found = {}
    if WINDOWS:
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        kernel.GetModuleFileNameW.restype = ctypes.c_uint32
        for name in CUDA_NAMES:
            handle = kernel.GetModuleHandleW(name)
            buffer = ctypes.create_unicode_buffer(32768)
            if handle and kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
                found[name] = buffer.value
        return found
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name in CUDA_NAMES:
            found[Path(fields[5]).name] = fields[5]
    return found


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def cli_output(
    cli: Path,
    model: Path,
    precision: str,
    cache: Path,
    sample: np.ndarray,
    temp: Path,
    output_shape: tuple[int, ...] | None = None,
    switches: tuple[str, ...] = (),
) -> np.ndarray:
    batch, _, height, width = sample.shape
    source, target = temp / "cli-input.raw", temp / "cli-output.raw"
    source.write_bytes(sample.tobytes())
    # The CLI resolves cuBLASLt and cuDNN through the platform search path.
    # Point it at the same files this process loaded.
    directories = os.pathsep.join(dict.fromkeys(str(Path(path).parent) for path in loaded_cuda_libraries().values()))
    variable = "PATH" if WINDOWS else "LD_LIBRARY_PATH"
    env = {**os.environ, variable: directories + os.pathsep + os.environ.get(variable, "")}
    subprocess.run(
        [
            str(cli),
            "run",
            *("--model", str(model), "--precision", precision, "--cache-dir", str(cache)),
            *("--batch", str(batch), "--height", str(height), "--width", str(width)),
            *("--input", str(source), "--output", str(target)),
            *switches,
        ],
        check=True,
        env=env,
    )
    return np.frombuffer(target.read_bytes(), dtype=sample.dtype).reshape(output_shape or sample.shape)


def check_precision(umzi: Any, args: argparse.Namespace, precision: str, temp: Path) -> dict[str, Any]:
    dtype = np.dtype("float16" if precision == "fp16" else "float32")
    cache = temp / f"cache-{precision}"
    options = umzi.SessionOptions(
        precision=precision,
        cache_dir=cache,
        dll_path=args.library,
        dll_dirs=args.cuda_dir,
    )
    rng = np.random.default_rng(700)
    rows: dict[str, dict[str, Any]] = {}
    samples = {}
    with umzi.InferenceSession(args.model, options) as session:
        library = session.library_path
        for shape in CASES:
            sample = rng.random(shape, dtype=np.float32).astype(dtype)
            (restored,) = session.run(None, {session.get_inputs()[0].name: sample})
            info = session.get_session_info()
            scale = session.scale_factor
            expected_shape = (shape[0], shape[1], shape[2] * scale, shape[3] * scale)
            rows["x".join(map(str, shape))] = {
                "dtype": str(restored.dtype),
                "finite": bool(np.isfinite(restored).all()),
                "shape_match": restored.shape == expected_shape and restored.dtype == dtype,
                "first_session_cache_status": info["cache_status"],
                "first_session_stages_ms": info["stages_ms"],
                "padded_shape": list(info["padded_shape"]),
                "output_sha256": digest(restored),
            }
            samples[shape] = sample, restored
    # A second session must read every tuned shape back from the disk cache.
    with umzi.InferenceSession(args.model, options) as session:
        for shape, (sample, first) in samples.items():
            (again,) = session.run(None, {"input": sample})
            row = rows["x".join(map(str, shape))]
            row["reopened_cache_status"] = session.get_session_info()["cache_status"]
            row["reopened_bitwise_match"] = np.array_equal(first, again)
    if args.cli is not None:
        for shape, (sample, first) in samples.items():
            expected = cli_output(args.cli, args.model, precision, cache, sample, temp, tuple(first.shape))
            rows["x".join(map(str, shape))]["bitwise_cli_match"] = np.array_equal(first, expected)
    return {"library_path": str(library), "cases": rows}


def failures(report: dict[str, Any], expected_library: Path | None) -> list[str]:
    problems = []
    for precision, result in report["precisions"].items():
        if expected_library is not None and Path(result["library_path"]) != expected_library.resolve():
            problems.append(f"{precision}: loaded {result['library_path']}, expected {expected_library}")
        for name, row in result["cases"].items():
            checks = {
                "finite": row["finite"],
                "shape_match": row["shape_match"],
                "first session miss": row["first_session_cache_status"] == "miss",
                "reopened hit": row["reopened_cache_status"] == "hit",
                "reopened_bitwise_match": row["reopened_bitwise_match"],
                "bitwise_cli_match": row.get("bitwise_cli_match", True),
            }
            problems += [f"{precision} {name}: {check} failed" for check, ok in checks.items() if not ok]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--cli", type=Path, help="Native CLI for the bitwise comparison")
    parser.add_argument("--library", type=Path, help="Pass this path as SessionOptions(dll_path=...)")
    parser.add_argument("--expect-library", type=Path, help="Fail unless sessions load this library")
    parser.add_argument(
        "--cuda-dir",
        type=Path,
        action="append",
        default=[],
        help="Pass this directory in SessionOptions(dll_dirs=...). Repeat as needed.",
    )
    parser.add_argument("--import-torch", action="store_true", help="Import PyTorch before umzi_native")
    parser.add_argument(
        "--expect-load-failure",
        action="store_true",
        help="Only check that opening a session reports missing CUDA libraries",
    )
    args = parser.parse_args()
    if args.import_torch:
        importlib.import_module("torch")
    umzi = importlib.import_module("umzi_native")
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "package": umzi.__file__,
        "model": str(args.model.resolve()),
        "torch_imported": "torch" in sys.modules,
        "cuda_dirs": [str(path) for path in args.cuda_dir],
    }
    problems = []
    if args.expect_load_failure:
        try:
            umzi.InferenceSession(args.model, umzi.SessionOptions(dll_path=args.library, dll_dirs=args.cuda_dir))
        except OSError as error:
            report["load_failure"] = str(error)
        else:
            problems.append("The session opened although its CUDA libraries should be unavailable")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Keep scratch files beside the report so the CLI writes where the caller can.
        with tempfile.TemporaryDirectory(prefix="session-check-", dir=args.output.parent) as directory:
            report["precisions"] = {
                precision: check_precision(umzi, args, precision, Path(directory)) for precision in ("mixed", "fp16")
            }
        problems = failures(report, args.expect_library)
    report["cuda_libraries"] = loaded_cuda_libraries()
    report["problems"] = problems
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cuda_libraries": report["cuda_libraries"], "problems": problems}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
