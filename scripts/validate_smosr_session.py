"""Check native SMoSR parity on a rectangular crop of the benchmark images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from benchmark_architectures import FAMILIES, ROOT, comparison, read_image
from reference.native_session import DeviceSession

from umzi_native import InferenceSession, SessionOptions
from umzi_native.exporter._loader import load_checkpoint


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    family = FAMILIES["smosr"]
    image = sorted((ROOT / "samples" / family.samples).glob("*.jpg"))[0]
    sample = read_image(image)[:, :, :65, :79].contiguous()
    reference, _ = load_checkpoint(ROOT / "models" / family.checkpoint)
    expected = reference.eval().cuda()(sample.cuda()).cpu().numpy()
    for precision in ("mixed", "fp16"):
        options = SessionOptions(precision=precision, dll_path=args.library)
        direct_options = SessionOptions(precision=precision, dll_path=args.library, enable_cuda_graph=False)
        with InferenceSession(args.model, options) as graph, InferenceSession(args.model, direct_options) as direct:
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
            print(f"{precision}: graph, direct, host, and alternate-stream outputs matched. {metrics}")


if __name__ == "__main__":
    main()
