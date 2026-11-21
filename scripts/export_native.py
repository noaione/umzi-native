"""Export a checkpoint as an Umzi model package or a legacy benchmark bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from umzi_native import GateRV3Model
from umzi_native._model import GateRV3Op as Op
from umzi_native.exporter import export_model
from umzi_native.exporter._gaterv3 import Schedule, build_schedule
from umzi_native.exporter._loader import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/1x_umzi_digital_decompress_gaterv3_1.pth"
OUTPUT_DIR = ROOT / "output/native"


class Activation(NamedTuple):
    index: int
    first: int
    last: int
    size: int

    def overlaps(self, other: Activation) -> bool:
        return self.first <= other.last and other.first <= self.last


def pack_activations(activations: list[Activation]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    placed: list[Activation] = []
    for tensor in sorted(activations, key=lambda item: (-item.size, item.first)):
        offset = 0
        conflicts = sorted((offsets[other.index], other.size) for other in placed if tensor.overlaps(other))
        for start, size in conflicts:
            if offset + tensor.size <= start:
                break
            offset = max(offset, start + size)
        offsets[tensor.index] = offset
        placed.append(tensor)

    for i, tensor in enumerate(activations):
        start = offsets[tensor.index]
        if start % 256:
            raise RuntimeError(f"Misaligned arena offset for tensor {tensor.index}")
        for other in activations[:i]:
            addr = offsets[other.index]
            if tensor.overlaps(other) and max(start, addr) < min(start + tensor.size, addr + other.size):
                raise RuntimeError(f"Arena overlap for tensors {tensor.index} and {other.index}")
    return offsets


def write_legacy_schedule(schedule: Schedule, path: Path, weight_bytes: int) -> int:
    last_use = list(range(-1, len(schedule.ops)))
    for index, op in enumerate(schedule.ops):
        for source in op[2:6]:
            if source >= 0:
                last_use[source] = index
    for index, (conv, silu) in enumerate(zip(schedule.ops, schedule.ops[1:], strict=False)):
        if conv[0] == Op.CONV and conv[9] == 3 and silu[0] == Op.SILU:
            # An optional fused launch reads the original convolution input
            # while writing the SiLU result.
            source = conv[2]
            last_use[source] = max(last_use[source], index + 1)
    activations = []
    for index, (h, w, c) in enumerate(schedule.shapes[1:-1], start=1):
        first = index - 1
        op = schedule.ops[first]
        # Residual + RMSNorm emits both outputs in the residual launch.
        if op[0] == Op.RMS and op[3] >= 0:
            first -= 1
        size = (h * w * c * 2 + 255) & ~255
        activations.append(Activation(index, first, last_use[index], size))
    offsets = pack_activations(activations)
    with path.open("wb") as file:
        file.write(
            struct.pack(
                "<4sIII",
                b"GTR3",
                len(schedule.shapes),
                len(schedule.ops),
                weight_bytes,
            )
        )
        for index, (h, w, c) in enumerate(schedule.shapes):
            file.write(struct.pack("<4i", h, w, c, offsets.get(index, -1)))
        for op in schedule.ops:
            file.write(struct.pack("<11i", *op))
    return max(offsets[tensor.index] + tensor.size for tensor in activations)


def export_legacy(checkpoint: Path, output: Path) -> None:
    model, state = load_checkpoint(checkpoint)
    with checkpoint.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as staging:
        directory = Path(staging)
        weights = directory / "weights.bin"
        schedule, _ = build_schedule(model, state, weights)
        arena = write_legacy_schedule(schedule, directory / "schedule.bin", weights.stat().st_size)

        artifacts = ("schedule.bin", "weights.bin")
        hashes = {}
        for name in artifacts:
            with (directory / name).open("rb") as file:
                hashes[name] = hashlib.file_digest(file, "sha256").hexdigest()
        (directory / "checkpoint.json").write_text(
            json.dumps({"checkpoint_sha256": digest, "weights_and_schedule": hashes}, indent=2) + "\n",
            encoding="utf-8",
        )
        for name in (*artifacts, "checkpoint.json"):
            (directory / name).replace(output / name)
    print(f"Checkpoint {digest}, {len(schedule.ops)} operations, {arena:,} arena bytes per tile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CHECKPOINT)
    parser.add_argument("--kernel", choices=("gaterv3",), default="gaterv3")
    parser.add_argument("--output", type=Path, help=".ummodel file, or bundle directory with --legacy")
    parser.add_argument("--legacy", action="store_true", help="Export the fixed schedule and weights separately")
    args = parser.parse_args()
    torch.set_num_threads(4)

    if args.legacy or (args.output is not None and args.output.suffix == ""):
        export_legacy(args.model, args.output or OUTPUT_DIR)
        return
    if args.output is not None:
        if args.output.suffix.lower() != ".ummodel":
            parser.error("The model package filename must end in .ummodel")
        info = export_model(args.model, args.output, kernel=args.kernel)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=OUTPUT_DIR) as staging:
            info = export_model(args.model, Path(staging) / "model.ummodel", kernel=args.kernel)
            assert isinstance(info.payload, GateRV3Model)
            name = "gaterv3.ummodel" if info.payload.alignment == 16 else f"{args.model.stem}.ummodel"
            output = info.path.replace(OUTPUT_DIR / name)
            info = replace(info, path=output)
    assert isinstance(info.payload, GateRV3Model)
    print(f"Exported {info.path} with {info.payload.operations} operations\nCheckpoint {info.payload.checkpoint}")


if __name__ == "__main__":
    main()
