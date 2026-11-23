"""Check native SMoSR parity on a rectangular crop of the benchmark images."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from benchmark_architectures import FAMILIES, ROOT, comparison, extract, grid, read_image, stress_inputs
from reference.native_session import DeviceSession

from umzi_native import InferenceSession, SessionOptions
from umzi_native.exporter._loader import load_checkpoint


def check_baseline(candidate: DeviceSession, baseline: DeviceSession, sample: torch.Tensor, label: str) -> None:
    sample = sample.to(device="cuda", dtype=candidate.dtype)
    np.testing.assert_array_equal(
        candidate(sample).cpu().numpy().view(np.uint8),
        baseline(sample).cpu().numpy().view(np.uint8),
        err_msg=label,
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--baseline-library", type=Path, help="Compare bitwise with the original native build")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    family = FAMILIES["smosr"]
    paths = sorted((ROOT / "samples" / family.samples).glob("*.jpg"))
    image = paths[0]
    sample = read_image(image)[:, :, :65, :79].contiguous()
    reference, _ = load_checkpoint(ROOT / "models" / family.checkpoint)
    expected = reference.eval().cuda()(sample.cuda()).cpu().numpy()
    for precision in ("mixed", "fp16"):
        options = SessionOptions(precision=precision, dll_path=args.library)
        direct_options = SessionOptions(precision=precision, dll_path=args.library, enable_cuda_graph=False)
        with ExitStack() as stack:
            graph = stack.enter_context(InferenceSession(args.model, options))
            direct = stack.enter_context(InferenceSession(args.model, direct_options))
            assert (graph.kernel, graph.scale_factor, graph.context) == ("smosr", 2, 0)
            captured, uncaptured = DeviceSession(graph), DeviceSession(direct)
            converted = sample.to(device="cuda", dtype=captured.dtype)
            output = captured(converted)
            actual = output.cpu().numpy()
            assert all(graph.get_session_info()["startup_device_used_bytes"].values())
            np.testing.assert_array_equal(uncaptured(converted).cpu().numpy(), actual)
            (host,) = graph.run(None, {"input": converted.cpu().numpy()})
            np.testing.assert_array_equal(host, actual)
            alternate = torch.cuda.Stream()
            with torch.cuda.stream(alternate):
                alternate.wait_stream(torch.cuda.default_stream())
                result = captured(converted)
            alternate.synchronize()
            np.testing.assert_array_equal(result.cpu().numpy(), actual)
            metrics = comparison(actual, expected)
            if not metrics["finite"] or (metrics["psnr_db"] is not None and metrics["psnr_db"] < 45):
                raise AssertionError(f"Poor SMoSR {precision} parity on {image.name}: {metrics}")
            try:
                graph.prepare((1, 3, 2, 5))
            except RuntimeError as error:
                if "height and width at least 3" not in str(error):
                    raise
            else:
                raise AssertionError("SMoSR accepted an input smaller than its reflection padding permits")
            if args.baseline_library:
                baseline_options = SessionOptions(precision=precision, dll_path=args.baseline_library)
                baseline = DeviceSession(stack.enter_context(InferenceSession(args.model, baseline_options)))
                for label, fixture in stress_inputs(family.tile, paths, family.batch):
                    check_baseline(captured, baseline, fixture, label)
                tiles = 0
                for path in paths:
                    rgb = read_image(path)
                    height, width = rgb.shape[-2:]
                    for top, left in grid(height, width, family.tile, family.overlap):
                        check_baseline(
                            captured, baseline, extract(rgb, top, left, family.tile), f"{path.name} ({top}, {left})"
                        )
                        tiles += 1
                print(f"{precision}: stress cases and {tiles} image tiles matched the native baseline bitwise")
            print(f"{precision}: graph, direct, host, and alternate-stream outputs matched. {metrics}")


if __name__ == "__main__":
    main()
