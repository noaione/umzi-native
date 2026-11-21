"""Check package rejection and a second kernel through the same Python API."""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from umzi_native import InferenceSession, SessionOptions
from umzi_native._model import GATERV3_FAMILY, HEADER_SIZE
from umzi_native.exporter._package import write_package

TEST_KERNEL = 0xFFFF0001


def rejected(path: Path, options: SessionOptions, message: str) -> None:
    try:
        with InferenceSession(path, options):
            pass
    except RuntimeError as error:
        if message in str(error):
            return
        raise
    raise RuntimeError(f"{path.name} was accepted, expected {message!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--test-library", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required to load the runtime for these checks")
    production = SessionOptions(dll_path=args.library)
    test_build = SessionOptions(dll_path=args.test_library)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        package = root / "test.ummodel"
        write_package(package, TEST_KERNEL, 1, {1: b"test"})
        rejected(package, production, "kernel family is unavailable")
        with InferenceSession(package, test_build) as session:
            assert session.kernel == "test_kernel" and session.scale_factor == 2 and session.context == 0
            assert session.get_inputs()[0].name == "source"
            assert session.get_outputs()[0].name == "enlarged"
            source = np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3)
            session.prepare(source.shape)
            (output,) = session.run(["enlarged"], {"source": source})
            np.testing.assert_array_equal(output, source.repeat(2, axis=2).repeat(2, axis=3))
            assert session.get_session_info()["padded_shape"] == source.shape

        original = package.read_bytes()
        package.write_bytes(b"GTR3PCK\0" + original[8:])
        rejected(package, production, "package magic")
        invalid = bytearray(original)
        struct.pack_into("<I", invalid, 8, 2)
        package.write_bytes(invalid)
        rejected(package, test_build, "package version")
        package.write_bytes(original[:-1])
        rejected(package, test_build, "package header")
        package.write_bytes(original[:-1] + b"!")
        rejected(package, test_build, "checksum mismatch")
        invalid = bytearray(original)
        struct.pack_into("<Q", invalid, HEADER_SIZE + 8, 0)
        invalid[32:64] = hashlib.sha256(invalid[:32] + invalid[64:]).digest()
        package.write_bytes(invalid)
        rejected(package, test_build, "section table")
        write_package(package, GATERV3_FAMILY, 2, {1: b"test"})
        rejected(package, production, "payload version")
    print("Kernel dispatch, independent output shape, and malformed package checks passed.")


if __name__ == "__main__":
    main()
