"""Profile GateRV3 dispatch groups with opt-in Vulkan timestamp queries."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from benchmark_vulkan import BATCH, ROOT, batch_sample, load_page

from umzi_native import InferenceSession, SessionOptions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--image", type=Path, default=ROOT / "samples/input/page-001.jpg")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, choices=(1, BATCH), default=1)
    parser.add_argument("--precision", choices=("fp32", "mixed", "fp16"), default="mixed")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--host-only", action="store_true", help="Time host calls with GPU timestamps disabled")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.reference and args.precision == "fp16":
        parser.error("The reference kernels support mixed and FP32")

    tiles = load_page(args.image).tiles
    sample = (batch_sample(tiles[:BATCH]) if args.batch == BATCH else tiles[0][2]).numpy()
    options = SessionOptions(
        backend="vulkan",
        device_id=args.device,
        precision=args.precision,
        dll_path=args.library,
        kernel_options={"reference": int(args.reference), "profile": int(not args.host_only)},
    )
    host_ms = []
    groups_ns: dict[str, list[int]] = defaultdict(list)
    valid_runs = 0
    with InferenceSession(args.model, options) as session:
        session.prepare(sample.shape)
        for _ in range(2):
            session.run(None, {"input": sample})
        info = session.get_session_info()
        diagnostics = info["diagnostics"]
        for _ in range(args.runs):
            started = time.perf_counter()
            session.run(None, {"input": sample})
            host_ms.append(1000 * (time.perf_counter() - started))
            diagnostics = session.get_session_info()["diagnostics"]
            if diagnostics.get("gpu_timestamps_valid"):
                valid_runs += 1
                for key, value in diagnostics.items():
                    if key.startswith("gpu_") and key.endswith("_ns"):
                        groups_ns[key[4:-3]].append(value)

    print(
        json.dumps(
            {
                "model": str(args.model),
                "device": info["kernel_message"],
                "batch": args.batch,
                "precision": args.precision,
                "reference": args.reference,
                "dispatches": diagnostics["dispatches"],
                "host_median_ms": statistics.median(host_ms),
                "timestamps_requested": not args.host_only,
                "valid_gpu_timestamp_runs": valid_runs,
                "gpu_group_median_ms": {name: statistics.median(times) / 1e6 for name, times in groups_ns.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
