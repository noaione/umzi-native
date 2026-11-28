"""Alternate MoSR session builds in one process and report device time."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from reference.native_session import DeviceSession

from umzi_native import InferenceSession, SessionOptions

PRECISIONS = {"fp16": torch.float16, "mixed": torch.float32, "fp32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--calls", type=int, default=3, help="Calls per timed sample")
    parser.add_argument("--precisions", nargs="+", default=["fp16", "mixed", "fp32"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("builds", nargs="+", help="NAME=path/to/umzi_native.dll")
    args = parser.parse_args()
    libraries = dict(build.split("=", 1) for build in args.builds)
    shape = (args.batch, 3, args.tile, args.tile)
    for precision in args.precisions:
        dtype = PRECISIONS[precision]
        sessions = {
            name: InferenceSession(
                args.model,
                SessionOptions(precision=precision, dll_path=library, cache_dir=args.cache_dir),
            )
            for name, library in libraries.items()
        }
        try:
            sample = torch.rand(shape, device="cuda", dtype=dtype)
            runs = {name: DeviceSession(session) for name, session in sessions.items()}
            for run in runs.values():
                run(sample)
            torch.cuda.synchronize()
            samples: dict[str, list[float]] = {name: [] for name in runs}
            for _ in range(args.rounds):
                for name, run in runs.items():
                    start = time.perf_counter()
                    for _ in range(args.calls):
                        run(sample)
                    torch.cuda.synchronize()
                    samples[name].append((time.perf_counter() - start) / args.calls * 1000)
            for name, values in samples.items():
                median = statistics.median(values)
                print(
                    f"{precision} {name}: median {median:.2f} ms per batch-{args.batch} call "
                    f"({median / args.batch:.2f} ms/tile), samples {[round(value, 1) for value in values]}"
                )
        finally:
            for session in sessions.values():
                session.close()


if __name__ == "__main__":
    main()
