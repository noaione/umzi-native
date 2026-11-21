"""Validate packed GateRV3 sessions across shapes, precisions and CUDA entry points."""

from __future__ import annotations

import argparse
import ctypes
import json
import tempfile
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from benchmark_native import CHECKPOINT, ROOT, Native, error_metrics, load_model
from torch.nn.attention import SDPBackend, sdpa_kernel

from umzi_native._native import IoABI

CASES = (
    (1, 128, 128),
    (16, 128, 128),
    (3, 192, 160),
    (7, 65, 79),
    (7, 79, 65),  # Same padded geometry as the previous case.
    (17, 128, 128),
    (1, 160, 112),
    (1, 1, 7),
    (1, 128, 128),
)


def reference_output(reference, sample):
    x = sample.cuda()
    h, w = x.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    # The fused Torch backend rejects the singleton latent token's strides.
    backend = sdpa_kernel(SDPBackend.MATH) if h + ph == w + pw == 16 else nullcontext()
    with backend:
        if ph < h and pw < w:
            return reference(x).cpu()
        for pad, gap, length in (((0, pw, 0, 0), pw, w), ((0, 0, 0, ph), ph, h)):
            if gap:
                x = F.pad(x, pad, mode="reflect" if gap < length else "replicate")
        return reference(x)[:, :, :h, :w].cpu()


def describe_and_check_buffers(runner, sample, shape):
    dll = runner.dll
    info = IoABI()
    info.struct_size = ctypes.sizeof(info)
    runner.check(dll.gt_session_describe(runner.handle, *shape, ctypes.byref(info)))
    batch, height, width = shape
    packed = sample.to(runner.dtype)
    assert (info.batch, info.channels, info.height, info.width) == (
        batch,
        3,
        height,
        width,
    )
    assert info.rgb_type == (1 if runner.dtype == torch.float32 else 2)
    assert info.input_bytes == info.output_bytes == packed.nbytes
    assert (
        dll.gt_session_run(
            runner.handle,
            *shape,
            packed.data_ptr(),
            packed.nbytes - 1,
            packed.data_ptr(),
            packed.nbytes,
        )
        == -1
    )
    assert b"require" in dll.gt_error()


def run_device(runner, sample, shape):
    x = sample.to(device="cuda", dtype=runner.dtype)
    y = torch.empty_like(x)
    runner.check(
        runner.dll.gt_session_enqueue(
            runner.handle,
            *shape,
            x.data_ptr(),
            x.nbytes,
            y.data_ptr(),
            y.nbytes,
            torch.cuda.current_stream().cuda_stream,
        )
    )
    return y.float().cpu()


def packed_setup(runner):
    setup = runner.setup()
    assert setup is not None
    return setup


def check_cache_recovery(bundle, model_file):
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        with ExitStack() as stack:
            runner = Native(bundle, 1, model_file=model_file, cache_dir=base)
            stack.callback(runner.close)
            assert packed_setup(runner)["cache_status"] == "miss"
        [entry] = base.glob("*.gtplan")
        entry.write_bytes(b"corrupt entry")
        with ExitStack() as stack:
            rebuilt = Native(bundle, 1, model_file=model_file, cache_dir=base)
            stack.callback(rebuilt.close)
            corrupt = packed_setup(rebuilt)
            assert corrupt["cache_status"] == "rebuilt"
            assert torch.isfinite(rebuilt.host(torch.zeros((1, 3, 128, 128)))).all()
        blocked = base / "file-for-directory"
        blocked.write_bytes(b"file")
        with ExitStack() as stack:
            unavailable = Native(bundle, 1, model_file=model_file, cache_dir=blocked)
            stack.callback(unavailable.close)
            blocked_setup = packed_setup(unavailable)
            assert blocked_setup["cache_status"] == "unavailable"
            assert torch.isfinite(unavailable.host(torch.zeros((1, 3, 128, 128)))).all()
    return {"corrupt_case": corrupt, "unwritable_case": blocked_setup}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "output/native-refactor")
    parser.add_argument(
        "--legacy-bundle", type=Path, default=ROOT / "output/native-projections"
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/native-refactor-validation"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_file = args.bundle / "gaterv3.gtmodel"
    reference, _ = load_model(args.checkpoint, device="cuda")
    result: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(dir=args.output) as directory:
        cache = Path(directory)
        for mode in ("mixed", "fp16"):
            rows = []
            with ExitStack() as stack:
                graph = Native(
                    args.bundle,
                    1,
                    precision=mode,
                    cache_dir=cache,
                    model_file=model_file,
                )
                stack.callback(graph.close)
                direct = Native(
                    args.bundle,
                    1,
                    precision=mode,
                    graph=False,
                    cache_dir=cache,
                    model_file=model_file,
                )
                stack.callback(direct.close)
                start_model_load = graph.session_info().model_load_ms
                first = None
                sample_generator = torch.Generator().manual_seed(700)
                for index, shape in enumerate(CASES):
                    batch, height, width = shape
                    g = (
                        torch.Generator().manual_seed(700)
                        if shape == CASES[0]
                        else sample_generator
                    )
                    x = torch.rand((batch, 3, height, width), generator=g)
                    describe_and_check_buffers(graph, x, shape)
                    expected = reference_output(reference, x)
                    output = graph.host(x).clone()
                    info = graph.session_info()
                    assert info.model_load_ms == start_model_load
                    assert (info.batch, info.padded_height, info.padded_width) == (
                        batch,
                        (height + 15) // 16 * 16,
                        (width + 15) // 16 * 16,
                    )
                    assert torch.equal(output, direct.host(x))
                    assert torch.equal(output, run_device(graph, x, shape))
                    setup = packed_setup(graph)
                    quality = error_metrics(expected, output)
                    psnr = quality["psnr_db"]
                    assert psnr is None or psnr > (70 if mode == "mixed" else 58)
                    row = {
                        "shape": list(shape),
                        "graph_direct_device_identical": True,
                        "cache_status": setup["cache_status"],
                        "shape_allocation_ms": setup["stages_ms"]["shape_allocation"],
                        "buffer_bytes": list(info.buffer_bytes),
                        "quality": quality,
                    }
                    if index == 0:
                        first = output
                    if index == len(CASES) - 1:
                        assert (
                            first is not None
                            and torch.equal(first, output)
                            and setup["cache_status"] == "hit"
                        )
                    if height == width == 128 and batch in (1, 16) and index < 2:
                        old = Native(
                            args.legacy_bundle, batch, precision=mode, cache_dir=cache
                        )
                        try:
                            row["old_fixed_comparison"] = error_metrics(
                                old.host(x), output
                            )
                        finally:
                            old.close()
                    print(mode, row, flush=True)
                    rows.append(row)
                assert rows[3]["shape_allocation_ms"] == rows[4]["shape_allocation_ms"]
                status = graph.dll.gt_session_prepare(graph.handle, 60, 512, 512)
                allocation_message = graph.dll.gt_error().decode()
                assert status == -2, (status, allocation_message)
                assert "bytes" in allocation_message
                small = torch.rand((1, 3, 33, 48), generator=sample_generator)
                assert torch.isfinite(graph.host(small)).all()
            result[mode] = {
                "cases": rows,
                "retried_after_allocation_failure": allocation_message,
            }
    result["cache_recovery"] = check_cache_recovery(args.bundle, model_file)
    (args.output / "shapes.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
