"""Export a GateRV3 checkpoint as an Umzi model package."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from umzi_native import GateRV3Model
from umzi_native.exporter import export_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/1x_umzi_digital_decompress_gaterv3_1.pth"
OUTPUT_DIR = ROOT / "output/native"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CHECKPOINT)
    parser.add_argument("--kernel", choices=("gaterv3",), default="gaterv3")
    parser.add_argument("--output", type=Path, help=".ummodel output path")
    args = parser.parse_args()
    torch.set_num_threads(4)

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
