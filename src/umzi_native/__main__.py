"""Command line entry point for peeking at Umzi model packages."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ._model import GateRV3Model, ModelInfo, peek_model


def _bytes(count: int) -> str:
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if count >= scale:
            return f"{count / scale:.2f} {unit} ({count:,} bytes)"
    return f"{count:,} bytes"


def _as_json(info: ModelInfo) -> dict[str, object]:
    return {
        "path": str(info.path),
        "supported": info.supported,
        "size": info.size,
        "version": info.version,
        "family": info.family,
        "family_name": info.family_name,
        "schema": info.schema,
        "digest": info.digest,
        "sections": [asdict(section) for section in info.sections],
        "payload": asdict(info.payload) if info.payload is not None else None,
        "problems": list(info.problems),
    }


def _summary(info: ModelInfo) -> str:
    family = info.family_name or f"kernel family {info.family}"
    state = "supported" if info.supported else "unsupported"
    lines = [f"{info.path}: {family} schema {info.schema}, {state}", f"  {_bytes(info.size)}, sha256 {info.digest}"]
    for section in info.sections:
        lines.append(f"  section {section.id} {section.name}: {_bytes(section.length)}")
    payload = info.payload
    if isinstance(payload, GateRV3Model):
        lines.append(
            f"  {payload.input_channels} -> {payload.output_channels} channels, x{payload.scale} scale, "
            f"{payload.alignment}-pixel padding, {payload.base_channels} base channels"
        )
        lines.append(
            f"  {payload.tensors} tensors, {payload.operations} operations, {_bytes(payload.weight_bytes)} of weights"
        )
        if payload.op_kinds:
            lines.append("  operations: " + ", ".join(f"{name} {count}" for name, count in payload.op_kinds.items()))
        if payload.checkpoint:
            lines.append(f"  checkpoint: {payload.checkpoint}")
    lines.extend(f"  problem: {problem}" for problem in info.problems)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the model information and return a non-zero status for unsupported packages."""
    parser = argparse.ArgumentParser(
        prog="python -m umzi_native",
        description="Read .ummodel packages and report whether this build supports them.",
    )
    parser.add_argument("models", metavar="MODEL", nargs="+", type=Path, help="packaged .ummodel file")
    parser.add_argument("--json", action="store_true", help="print a JSON array instead of a summary")
    args = parser.parse_args(argv)
    reports: list[dict[str, object]] = []
    status = 0
    for model in args.models:
        try:
            info = peek_model(model)
        except (OSError, ValueError) as error:
            status = 1
            if args.json:
                reports.append({"path": str(model.resolve()), "supported": False, "error": str(error)})
            else:
                print(f"{model}: {error}", file=sys.stderr)
            continue
        status |= not info.supported
        if args.json:
            reports.append(_as_json(info))
        else:
            print(_summary(info))
    if args.json:
        print(json.dumps(reports, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
