"""Alternate MoESR session builds in one process and report batch device time.

Laptop GPUs drift with temperature, so comparing builds in separate runs is
unreliable. This script opens every build at once and interleaves short timed
batches of identical random tiles, so each build sees the same thermal state.

Example:
    python scripts/ab_moesr_builds.py --model output/moesr-session/validated-p2/moesr.ummodel \
        base=output/moesr-session/opt-win/umzi_native.dll new=output/moesr-session/p2-win/umzi_native.dll
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from reference.native_session import DeviceSession

from umzi_native import InferenceSession, SessionOptions

PRECISIONS = {"fp16": torch.float16, "mixed": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--calls", type=int, default=4, help="Calls per timed sample")
    parser.add_argument("builds", nargs="+", help="NAME=path/to/umzi_native.dll")
    args = parser.parse_args()
    libraries = dict(build.split("=", 1) for build in args.builds)
    shape = (args.batch, 3, args.tile, args.tile)
    for precision, dtype in PRECISIONS.items():
        sessions = {
            name: InferenceSession(args.model, SessionOptions(precision=precision, dll_path=library))
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
                    f"{precision} {name}: median {median:.1f} ms per batch-{args.batch} call "
                    f"({median / args.batch:.1f} ms/tile), samples {[round(value, 1) for value in values]}"
                )
        finally:
            for session in sessions.values():
                session.close()


if __name__ == "__main__":
    main()
