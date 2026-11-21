"""Read and export Umzi model packages from the command line."""

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


def _read(args: argparse.Namespace) -> int:
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


def _export(args: argparse.Namespace) -> int:
    from .exporter import export_model

    try:
        info = export_model(args.checkpoint, args.output, kernel=args.kernel)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print(f"{args.checkpoint}: {error}", file=sys.stderr)
        return 1
    print(_summary(info))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the requested model command and return its exit status."""
    parser = argparse.ArgumentParser(prog="python -m umzi_native", description="Read and export .ummodel packages.")
    commands = parser.add_subparsers(dest="command", required=True)

    read = commands.add_parser("read", help="inspect one or more .ummodel packages")
    read.add_argument("models", metavar="MODEL", nargs="+", type=Path, help="packaged .ummodel file")
    read.add_argument("--json", action="store_true", help="print a JSON array instead of a summary")
    read.set_defaults(handler=_read)

    export = commands.add_parser(
        "export", help="pack a SafeTensors or PyTorch checkpoint (requires the exporter extra)"
    )
    export.add_argument("checkpoint", type=Path, help=".safetensors, .pth, .pt, or .ckpt state dictionary")
    export.add_argument("-o", "--output", type=Path, required=True, help="destination .ummodel file")
    export.add_argument("--kernel", help="assert the detected native kernel family, such as gaterv3")
    export.set_defaults(handler=_export)

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in ("read", "export", "-h", "--help"):
        arguments.insert(0, "read")  # Preserve the original positional package reader.
    args = parser.parse_args(arguments)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
