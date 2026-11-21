"""Check native GateRV3 fidelity and compare synchronized execution paths."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark import IMAGE_EXTS, ROOT, image_tensor, load_model, patches, tensor_image
from build_native import libraries
from export_native import Op
from PIL import Image

CHECKPOINT = ROOT / "models/1x_umzi_digital_decompress_gaterv3_1.pth"
SHAPE = (3, 128, 128)
METRIC = Path(r"C:\Tools\imaging\ssimulacra2.exe")


class Native:
    def __init__(self, bundle, batch, graph=True, precision="mixed"):
        cuda, cudnn = libraries()
        self.dll_dirs = []
        try:
            for path in (cuda / "bin/x86_64", cudnn / "bin"):
                self.dll_dirs.append(os.add_dll_directory(str(path)))
            self.dll = ctypes.CDLL(str((bundle / "gaterv3.dll").resolve()))
            fp16 = precision == "fp16"
            create = self.dll.gt_create_fp16 if fp16 else self.dll.gt_create
            create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            create.restype = ctypes.c_void_p
            self.dll.gt_error.restype = ctypes.c_char_p
            self.dll.gt_destroy.argtypes = [ctypes.c_void_p]
            self.dll.gt_destroy.restype = None
            self.dll.gt_input.argtypes = [ctypes.c_void_p]
            self.dll.gt_input.restype = ctypes.c_uint64
            self.run = self.dll.gt_run_fp16 if fp16 else self.dll.gt_run
            self.run.argtypes = [ctypes.c_void_p] * 3
            self.dll.gt_enqueue.argtypes = [ctypes.c_void_p] * 2
            self.handle = create(str(bundle.resolve()).encode(), batch, graph)
            if not self.handle:
                self.check(-1)
        except BaseException:
            for directory in self.dll_dirs:
                directory.close()
            raise
        self.dtype = torch.float16 if fp16 else torch.float32
        self.host_output = torch.empty((batch, *SHAPE), dtype=self.dtype)
        self.device_input = self.dll.gt_input(self.handle)

    def check(self, status):
        if status:
            raise RuntimeError(self.dll.gt_error().decode())

    def close(self):
        self.dll.gt_destroy(self.handle)
        for directory in self.dll_dirs:
            directory.close()

    def host(self, x):
        if (
            x.shape != self.host_output.shape
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or not x.is_contiguous()
        ):
            raise ValueError(
                "Native benchmark input must be contiguous FP32 NCHW at the runner's batch size"
            )
        packed = x.to(dtype=self.dtype)
        self.check(
            self.run(self.handle, packed.data_ptr(), self.host_output.data_ptr())
        )
        return self.host_output.float()

    def prepare_device(self, x):
        from cuda.bindings import driver

        packed = x.to(dtype=self.dtype)
        torch.cuda.synchronize()
        (status,) = driver.cuMemcpyDtoD(
            self.device_input, packed.data_ptr(), packed.numel() * packed.element_size()
        )
        if status:
            raise RuntimeError(f"Native device input copy failed: {status}")

        def run():
            self.check(
                self.dll.gt_enqueue(
                    self.handle, torch.cuda.current_stream().cuda_stream
                )
            )

        return run

    def memory_values(self, symbol, keys):
        query = getattr(self.dll, symbol, None)
        if query is None:
            return None
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        query.restype = None
        sizes = (ctypes.c_uint64 * 4)()
        query(self.handle, sizes)
        return dict(zip(keys, sizes, strict=True))

    def profile(self, schedule):
        self.dll.gt_operation_count.argtypes = [ctypes.c_void_p]
        self.dll.gt_profile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        count = self.dll.gt_operation_count(self.handle)
        times = np.zeros(count, dtype=np.float32)
        self.check(self.dll.gt_profile(self.handle, times.ctypes.data, count))
        table = np.fromfile(schedule, dtype="<i4", offset=16)
        shapes = table[: 4 * (count + 1)].reshape(-1, 4)
        ops = table[4 * (count + 1) :].reshape(-1, 11)
        totals = defaultdict(float)
        rows = []
        for entry, ms in zip(ops, times, strict=True):
            kind, output, source = map(int, entry[:3])
            label = Op(kind).name.lower()
            if kind == Op.CONV:
                label = f"conv{int(entry[9])}x{int(entry[9])}"
            totals[label] += float(ms)
            rows.append(
                {
                    "operation": label,
                    "input_shape": shapes[source, :3].tolist(),
                    "output_shape": shapes[output, :3].tolist(),
                    "milliseconds": float(ms),
                }
            )
        return {"totals_ms": dict(totals), "operations": rows}


def error_metrics(reference, result):
    if not torch.isfinite(result).all():
        raise RuntimeError("Nonfinite model output")
    delta = reference.double() - result.double()
    mse = float(delta.square().mean())
    return {
        "mse": mse,
        "psnr_db": -10 * np.log10(mse) if mse else None,
        "max_absolute_error": float(delta.abs().max()),
    }


def batch_input(group, batch):
    inputs = [patch for _, _, patch in group]
    inputs.extend([inputs[-1]] * (batch - len(inputs)))
    return torch.cat(inputs)


def quality(args, images):
    reference, _ = load_model(args.model, device="cuda")
    rows = []
    scores = []
    with ExitStack() as runners:
        native = Native(args.bundle, 16, precision=args.precision)
        runners.callback(native.close)
        single = Native(args.bundle, 1, precision=args.precision)
        runners.callback(single.close)
        for name, image in images:
            result = torch.empty_like(image)
            expected = torch.empty_like(image)
            tiles = patches(image, 128)
            height, width = image.shape[-2:]
            for start in range(0, len(tiles), 16):
                group = tiles[start : start + 16]
                x = batch_input(group, 16)
                y = native.host(x)
                ref = reference(x.cuda()).cpu()
                for i, (top, left, _) in enumerate(group):
                    h, w = min(128, height - top), min(128, width - left)
                    result[0, :, top : top + h, left : left + w] = y[i, :, :h, :w]
                    expected[0, :, top : top + h, left : left + w] = ref[i, :, :h, :w]
            native_file = args.output / f"{Path(name).stem}-native.png"
            ref_file = args.output / f"{Path(name).stem}-fp32.png"
            image_tensor(result).save(native_file)
            image_tensor(expected).save(ref_file)
            row = {
                "image": name,
                "tiles": len(tiles),
                **error_metrics(expected, result),
            }
            if METRIC.is_file():
                score = subprocess.check_output(
                    [str(METRIC), str(ref_file), str(native_file)], text=True
                )
                values = re.findall(r"-?\d+(?:\.\d+)?", score)
                score = float(values[-1])
                row["ssimulacra2"] = score
                scores.append(score)
            rows.append(row)
            print(row, flush=True)

        generator = torch.Generator().manual_seed(700)
        checker = torch.arange(128)
        checker = (checker[:, None] + checker) % 2
        stress = {
            "black": torch.zeros((1, *SHAPE)),
            "white": torch.ones((1, *SHAPE)),
            "checkerboard": checker.float().expand(1, 3, 128, 128).contiguous(),
            "random": torch.rand((1, *SHAPE), generator=generator),
            "page_tile": patches(images[0][1], 128)[0][2],
        }
        stress_rows = []
        for label, sample in stress.items():
            sample = sample.contiguous()
            ref = reference(sample.cuda()).cpu()
            out = single.host(sample)
            tiled = native.host(sample.expand(16, -1, -1, -1).contiguous())[:1]
            row = {
                "sample": label,
                **error_metrics(ref, out),
                "batch16_first_tile_vs_batch1": error_metrics(out, tiled),
            }
            stress_rows.append(row)
            print(row, flush=True)
    pixels = [image.numel() for _, image in images]
    mse = sum(
        row["mse"] * count for row, count in zip(rows, pixels, strict=True)
    ) / sum(pixels)
    return {
        "pages": rows,
        "aggregate_psnr_db": -10 * np.log10(mse) if mse else None,
        "ssimulacra2": {"mean": statistics.mean(scores), "minimum": min(scores)}
        if scores
        else None,
        "batch1": stress_rows,
    }


def ort_session(onnx_path, profile_dir):
    import onnxruntime as ort
    import onnxruntime_ep_nv_tensorrt_rtx as rtx

    ort.disable_telemetry_events()
    ep = rtx.get_ep_name()
    if not any(device.ep_name == ep for device in ort.get_ep_devices()):
        ort.register_execution_provider_library(ep, rtx.get_library_path())

    def make_session(profile):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices(
            [d for d in ort.get_ep_devices() if d.ep_name == ep], {}
        )
        if profile:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_dir / onnx_path.stem)
        session = ort.InferenceSession(
            str(onnx_path), sess_options=options, enable_fallback=0
        )
        if ep not in session.get_providers():
            raise RuntimeError(f"Inactive RTX provider: {session.get_providers()}")
        return session

    profiled = make_session(True)
    input_shape = profiled.get_inputs()[0].shape
    profiled.run(None, {"input": np.zeros(input_shape, dtype=np.float16)})
    events = json.loads(Path(profiled.end_profiling()).read_text(encoding="utf-8"))
    nodes = [
        event for event in events if event.get("cat") == "Node" and event.get("dur", 0)
    ]
    placement = sorted(
        {event.get("args", {}).get("provider", "") for event in nodes} - {""}
    )
    if placement != [ep]:
        raise RuntimeError(f"Unexpected RTX node placement {placement}")
    return make_session(False), placement


def measure(call, repeats, *args):
    times = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        call(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - started) / 1e6)
    return times


def single_tile_timings(calls, images):
    # Alternate blocks of real tiles to reduce bias from clock and load drift.
    names = list(calls)
    reports = {name: {"per_image": [], "per_tile": []} for name in names}
    phase = 0
    for image_name, image in images:
        items = patches(image, 128)
        page = {name: [] for name in names}
        for start in range(0, len(items), 16):
            order = names if phase % 2 == 0 else list(reversed(names))
            phase += 1
            group = items[start : start + 16]
            for name in order:
                for index, (top, left, sample) in enumerate(group, start):
                    seconds = measure(calls[name], 1, sample.contiguous())[0] / 1000
                    page[name].append(
                        {
                            "image": image_name,
                            "tile_index": index,
                            "top": top,
                            "left": left,
                            "seconds": seconds,
                        }
                    )
        for name, rows in page.items():
            reports[name]["per_tile"].extend(rows)
            reports[name]["per_image"].append(
                {
                    "image": image_name,
                    "tiles": len(items),
                    "source_pixels": image.shape[-2] * image.shape[-1],
                    "average_tile_ms": statistics.mean(row["seconds"] for row in rows)
                    * 1000,
                }
            )
    return {
        name: {
            **row,
            "average_tile_ms": statistics.mean(
                sample["seconds"] for sample in row["per_tile"]
            )
            * 1000,
        }
        for name, row in reports.items()
    }


def batch16_page_timings(call, images):
    rows = []
    for name, image in images:
        items = patches(image, 128)
        started = time.perf_counter()
        for offset in range(0, len(items), 16):
            call(batch_input(items[offset : offset + 16], 16))
        torch.cuda.synchronize()
        image_seconds = time.perf_counter() - started
        pixels = image.shape[-2] * image.shape[-1]
        rows.append(
            {
                "image": name,
                "tiles": len(items),
                "source_pixels": pixels,
                "batch16_seconds": image_seconds,
                "batch16_megapixels_per_second": pixels / image_seconds / 1e6,
            }
        )
    seconds = sum(row["batch16_seconds"] for row in rows)
    return {
        "per_image": rows,
        "batch16_seconds": seconds,
        "batch16_megapixels_per_second": sum(row["source_pixels"] for row in rows)
        / seconds
        / 1e6,
    }


def speed(args, images, batch, report, save):
    host_input = batch_input(patches(images[0][1], 128)[:batch], batch)
    device_input = host_input.cuda()
    device_half = device_input.half()
    torch.cuda.synchronize()
    with ExitStack() as stack:

        def native_case(bundle, precision, graph=True):
            instance = Native(bundle, batch, graph=graph, precision=precision)
            stack.callback(instance.close)
            return instance

        native = native_case(args.bundle, args.precision)
        cases = {"native_graph": (native.prepare_device(device_input), native.host)}
        if args.control_bundle:
            control = native_case(args.control_bundle, "mixed")
            cases["native_control"] = (
                control.prepare_device(device_input),
                control.host,
            )
        fp16 = None
        if args.compare_fp16:
            fp16 = native_case(args.bundle, "fp16")
            cases["native_fp16"] = (fp16.prepare_device(device_input), fp16.host)
        elif not args.control_bundle:
            eager = native_case(args.bundle, args.precision, graph=False)
            cases["native_eager"] = (eager.prepare_device(device_input), eager.host)
        if "torch" in args.baselines:
            eager, _ = load_model(args.model, device="cuda", dtype=torch.float16)

            def torch_calls(model):
                return (
                    lambda: model(device_half),
                    lambda x: (
                        model(x.to(device="cuda", dtype=torch.float16)).float().cpu()
                    ),
                )

            cases["torch_eager"] = torch_calls(eager)
            if "inductor" in args.baselines:
                compiled = torch.compile(eager, mode="reduce-overhead", dynamic=False)
                cases["torch_inductor"] = torch_calls(compiled)
        if "rtx" in args.baselines:
            graph = args.baseline / f"gaterv3-fp16-t128-b{batch}.onnx"
            session, placement = ort_session(graph, args.output)
            report["rtx_provider_placement"] = placement
            report["rtx_graph_sha256"] = digest(graph)
            binding = session.io_binding()
            device_output = torch.empty_like(device_half)
            binding.bind_input(
                "input",
                "cuda",
                0,
                np.float16,
                tuple(device_half.shape),
                device_half.data_ptr(),
            )
            binding.bind_output(
                "output",
                "cuda",
                0,
                np.float16,
                tuple(device_half.shape),
                device_output.data_ptr(),
            )
            cases["trt_rtx"] = (
                lambda: session.run_with_iobinding(binding),
                lambda x: torch.from_numpy(
                    np.asarray(
                        session.run(None, {"input": x.numpy().astype(np.float16)})[0],
                        dtype=np.float32,
                    )
                ),
            )
        for name, (device, host) in cases.items():
            print(f"Warming batch {batch} {name}", flush=True)
            device()
            host(host_input)
            measure(device, 5)
            measure(host, 5, host_input)
        rounds = {name: {"device": [], "host": []} for name in cases}
        for index in range(args.rounds):
            order = list(cases) if index % 2 == 0 else list(reversed(cases))
            for name in order:
                device, host = cases[name]
                rounds[name]["device"].append(measure(device, args.repeats))
                rounds[name]["host"].append(measure(host, args.repeats, host_input))
        for name, boundaries in rounds.items():
            report[name] = {}
            for boundary, samples in boundaries.items():
                pooled = [sample for group in samples for sample in group]
                median = statistics.median(pooled)
                report[name][boundary] = {
                    "median_ms": median,
                    "mean_ms": statistics.mean(pooled),
                    "p95_ms": float(np.percentile(pooled, 95)),
                    "megapixels_per_second_at_median": batch
                    * 128
                    * 128
                    / median
                    / 1000,
                    "round_samples_ms": samples,
                }
            print(
                f"Batch {batch} {name} median ms "
                f"device={report[name]['device']['median_ms']:.3f} "
                f"host={report[name]['host']['median_ms']:.3f}",
                flush=True,
            )
        save()
        if args.profile:
            report["native_profile"] = native.profile(args.bundle / "schedule.bin")
            print(
                f"Batch {batch} native profile {report['native_profile']['totals_ms']}",
                flush=True,
            )
            save()
            if fp16 is not None:
                report["native_fp16_profile"] = fp16.profile(
                    args.bundle / "schedule.bin"
                )
                save()
        if not args.skip_pages:
            calls = {
                name: host
                for name, (_, host) in cases.items()
                if name != "native_eager"
            }
            if batch == 1:
                report["tile_timing_order"] = "alternating runner order every 16 tiles"
                for name, pages in single_tile_timings(calls, images).items():
                    report[name]["pages"] = pages
                save()
                print("Batch 1 paired tile timings saved", flush=True)
            else:
                for name, host in calls.items():
                    report[name]["pages"] = batch16_page_timings(host, images)
                    save()
                    print(f"Batch {batch} {name} page timings saved", flush=True)


def memory(args, batch):
    # Initialize the CUDA context before the baseline and sample outside timed runs.
    values = {}
    sample = torch.zeros((batch, *SHAPE), dtype=torch.float32)
    torch.cuda.init()
    runners = {"native_graph": (args.bundle, args.precision)}
    if args.control_bundle:
        runners["native_control"] = (args.control_bundle, "mixed")
    if args.compare_fp16:
        runners["native_fp16"] = (args.bundle, "fp16")

    def used():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return total - free, total

    for label, (bundle, precision) in runners.items():
        torch.cuda.empty_cache()
        before, capacity = used()
        started = time.perf_counter()
        native = Native(bundle, batch, precision=precision)
        initialization_ms = (time.perf_counter() - started) * 1000
        try:
            captured, _ = used()
            for _ in range(8):
                native.host(sample)
            warmed, _ = used()
            buffers = native.memory_values(
                "gt_buffer_sizes", ("weights", "activations", "workspace", "rgb_io")
            )
            startup = native.memory_values(
                "gt_startup_memory",
                ("context", "tuning_workspace", "tuned", "captured"),
            )
        finally:
            native.close()
        released, _ = used()
        row = {
            "initialization_wall_ms": initialization_ms,
            "device_capacity_bytes": capacity,
            "device_used_before_bytes": before,
            "device_used_captured_bytes": captured,
            "device_used_warmed_bytes": warmed,
            "device_used_released_bytes": released,
            "device_used_warmed_delta_bytes": warmed - before,
            "runner_buffers_bytes": buffers,
            "startup_device_used_bytes": startup,
            "startup_sampled_max_delta_bytes": max(startup.values())
            - startup["context"]
            if startup
            else None,
        }
        if buffers:
            buffers["total"] = sum(buffers.values())
        print(f"Batch {batch} {label} VRAM {row}", flush=True)
        values[label] = row
    return values


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CHECKPOINT)
    parser.add_argument("--input", type=Path, default=ROOT / "samples/input")
    parser.add_argument("--bundle", type=Path, default=ROOT / "output/native")
    parser.add_argument("--control-bundle", type=Path)
    parser.add_argument("--precision", choices=("mixed", "fp16"), default="mixed")
    parser.add_argument(
        "--compare-fp16",
        action="store_true",
        help="Include full FP16 in the timing and memory comparisons",
    )
    parser.add_argument("--baseline", type=Path, default=ROOT / "output/baseline")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/native-validation"
    )
    parser.add_argument(
        "--suite", choices=("quality", "speed", "memory", "all"), default="all"
    )
    parser.add_argument(
        "--baselines",
        choices=("torch", "inductor", "rtx"),
        nargs="*",
        default=["torch", "inductor", "rtx"],
    )
    parser.add_argument(
        "--batch", type=int, choices=(1, 16), nargs="+", default=[1, 16]
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect an instrumented native op profile",
    )
    parser.add_argument(
        "--skip-pages", action="store_true", help="Time repeated batches only"
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("Rounds and repeats must be positive")
    if "inductor" in args.baselines and "torch" not in args.baselines:
        parser.error("Include torch when benchmarking inductor")
    if args.compare_fp16 and args.precision != "mixed":
        parser.error("--compare-fp16 pairs the mixed runner with full FP16")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    images = [
        (p.name, tensor_image(Image.open(p)))
        for p in sorted(args.input.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not images:
        parser.error(f"No images found in {args.input}")

    checkpoint_hash = digest(args.model)

    def checked_bundle(bundle):
        manifest = json.loads((bundle / "checkpoint.json").read_text(encoding="utf-8"))
        hashes = {
            name: digest(bundle / name)
            for name in ("gaterv3.dll", "kernels.cubin", "schedule.bin", "weights.bin")
        }
        if manifest["checkpoint_sha256"] != checkpoint_hash or any(
            hashes[name] != manifest["weights_and_schedule"][name]
            for name in ("schedule.bin", "weights.bin")
        ):
            parser.error(
                f"Native schedule or weights do not match the checkpoint manifest in {bundle}"
            )
        return hashes

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "checkpoint_sha256": checkpoint_hash,
        "bundle_sha256": checked_bundle(args.bundle),
        "precision": args.precision,
    }
    if args.control_bundle:
        report["control_sha256"] = checked_bundle(args.control_bundle)

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    if args.suite in ("all", "memory"):
        report["memory"] = {}
        for batch in args.batch:
            report["memory"][str(batch)] = memory(args, batch)
            save()
    if args.suite in ("all", "quality"):
        report["quality"] = quality(args, images)
        save()
    if args.suite in ("all", "speed"):
        report["speed"] = {}
        for batch in args.batch:
            report["speed"][str(batch)] = {}
            speed(args, images, batch, report["speed"][str(batch)], save)
            save()


if __name__ == "__main__":
    main()
