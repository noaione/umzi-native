"""Exercise GateRV3 Vulkan shape changes, replay, and input validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reference.model import load_model

from umzi_native import InferenceSession, SessionOptions, vulkan_devices


def require_error(action, expected, message: str) -> None:
    try:
        action()
    except expected as error:
        if message in str(error):
            return
        raise
    raise AssertionError(f"Expected an error containing {message!r}")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--device", type=int, action="append")
    parser.add_argument("--torch-device", choices=("cuda", "mps", "cpu"), default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.torch_device)
    torch_model, _ = load_model(args.checkpoint, device=device, dtype=torch.float32)
    random = np.random.default_rng(23011)
    checker = (np.indices((65, 79)).sum(axis=0) % 2).astype(np.float32)
    patterns = {
        "random_square": random.random((1, 3, 128, 128), dtype=np.float32),
        "random_rectangle": random.random((2, 3, 41, 65), dtype=np.float32),
        "flat_odd": np.full((1, 3, 65, 79), 0.5, dtype=np.float32),
        "checker_odd": np.broadcast_to(checker, (1, 3, 65, 79)).copy(),
        "tiny": random.random((1, 3, 1, 3), dtype=np.float32),
    }
    # The native padding contract uses edge replication when reflection cannot cover the padding.
    # Apply that contract before the reference forward for the tiny case.
    refs = {}
    for name, sample in patterns.items():
        value = torch.from_numpy(sample).to(device)
        if name == "tiny":
            pad = torch_model.pad
            value = torch.nn.functional.pad(value, (0, pad - 3, 0, pad - 1), mode="replicate")
        refs[name] = torch_model(value).cpu().numpy()[..., : sample.shape[-2], : sample.shape[-1]]
    rows = []
    devices = vulkan_devices(args.library)
    if not devices:
        raise RuntimeError("No Vulkan compute GPU was found")
    square = patterns["random_square"]
    for index in args.device or list(range(len(devices))):
        precision_outputs = []
        precisions = ["fp32", "mixed"]
        if devices[index]["shader_float16"]:
            precisions.append("fp16")
        for precision in precisions:
            options = SessionOptions(backend="vulkan", precision=precision, device_id=index, dll_path=args.library)
            with InferenceSession(args.model, options) as session:
                session.prepare(square.shape)
                session.prepare(square.shape)
                results = {}
                for name, sample in patterns.items():
                    (out,) = session.run(None, {"input": sample})
                    assert out.shape == sample.shape and out.dtype == np.float32
                    assert np.isfinite(out).all(), name
                    delta = out.astype(np.float64) - refs[name]
                    mse = float(np.mean(np.square(delta)))
                    quality = float(-10 * np.log10(mse)) if mse else None
                    assert quality is None or quality >= 55, (index, precision, name, quality)
                    rows.append({
                        "driver": devices[index]["driver_name"],
                        "precision": precision,
                        "input": name,
                        "psnr_db": quality,
                    })
                    results[name] = out
                require_error(lambda: session.prepare((1, 1, 8, 8)), RuntimeError, "RGB NCHW")
                require_error(lambda: session.run(None, {"input": square.astype(np.float16)}), TypeError, "dtype")
                (replay,) = session.run(None, {"input": square})
                np.testing.assert_array_equal(replay, results["random_square"])
            require_error(lambda: session.prepare(square.shape), RuntimeError, "closed")
            with InferenceSession(args.model, options) as reopened:
                (replay,) = reopened.run(None, {"input": square})
                np.testing.assert_array_equal(replay, results["random_square"])
                precision_outputs.append(replay)
        assert not np.array_equal(precision_outputs[0], precision_outputs[1]), (
            "Mixed mode lost its FP16 rounding boundaries"
        )
        if len(precision_outputs) == 3:
            assert not np.array_equal(precision_outputs[1], precision_outputs[2]), (
                "FP16 mode and mixed mode returned the same seeded output"
            )
    with TemporaryDirectory() as temp:
        invalid = Path(temp) / "invalid.ummodel"
        invalid.write_bytes(bytes(64))
        require_error(
            lambda: InferenceSession(invalid, SessionOptions(backend="vulkan", dll_path=args.library)),
            RuntimeError,
            "package magic",
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
