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
from export_native import (
    MODEL_FORMAT_VERSION,
    MODEL_HEADER,
    MODEL_MAGIC,
    MODEL_METADATA,
    MODEL_SECTION,
    Op,
)
from PIL import Image
from rtx_session import create_rtx_session

from umzi_native._native import (
    CACHE_STATUSES,
    KERNEL_OPTIONS,
    OPTIMIZATION_STATS,
    OptionsABI,
    load_session_library,
    read_session_info,
    session_setup,
)

CHECKPOINT = ROOT / "models/1x_umzi_digital_decompress_gaterv3_1.pth"
SHAPE = (3, 128, 128)
METRIC = Path(r"C:\Tools\imaging\ssimulacra2.exe")


def packed_schedule(path):
    length = path.stat().st_size
    with path.open("rb") as file:
        magic, version, count, _ = MODEL_HEADER.unpack(file.read(MODEL_HEADER.size))
        if (magic, version, count) != (MODEL_MAGIC, MODEL_FORMAT_VERSION, 4):
            raise ValueError("Unsupported packed GateRV3 format")
        position = MODEL_HEADER.size + count * MODEL_SECTION.size
        table = file.read(count * MODEL_SECTION.size)
        sections = {}
        for index, (kind, reserved, offset, size) in enumerate(MODEL_SECTION.iter_unpack(table), start=1):
            if kind != index or reserved or offset != position or size > length - position:
                raise ValueError("Invalid packed GateRV3 sections")
            sections[kind] = (offset, size)
            position += size
        if (
            len(sections) != count
            or position != length
            or sections[1][1] != MODEL_METADATA.size
            or sections[4][1] != 32
        ):
            raise ValueError("Invalid packed GateRV3 section sizes")
        file.seek(sections[1][0])
        *_, tensors, operations = MODEL_METADATA.unpack(file.read(MODEL_METADATA.size))
        if not 2 <= operations <= 2000 or tensors != operations + 1 or sections[2][1] != tensors * 8 + operations * 44:
            raise ValueError("Invalid packed GateRV3 schedule length")
        file.seek(sections[2][0])
        specs = np.frombuffer(file.read(tensors * 8), dtype="<i4").reshape(-1, 2)
        ops = np.frombuffer(file.read(operations * 44), dtype="<i4").reshape(-1, 11)
        file.seek(sections[4][0])
        return specs, ops, file.read(32).hex()


def enabled_optimizations(args):
    return tuple(name for name in KERNEL_OPTIONS if getattr(args, name.replace("-", "_")))


def optimization_candidates(args):
    base = enabled_optimizations(args)
    for mode in ("mixed", "fp16") if args.compare_fp16 else (args.precision,):
        for name in args.compare_optimization:
            if name in base or (name == "fp16-qkv" and mode == "fp16"):
                continue
            yield f"native_{mode}_{name.replace('-', '_')}", mode, (*base, name)


class Native:
    def __init__(
        self,
        bundle,
        batch,
        graph=True,
        precision="mixed",
        cache_dir=None,
        fuse_silu=False,
        optimizations=(),
        model_file=None,
    ):
        self.dll_dirs = ExitStack()
        self.handle = None
        self.model_file = model_file
        self.bundle = bundle
        fp16 = precision == "fp16"
        self.dtype = torch.float16 if fp16 else torch.float32
        flags = int(graph) | (2 if fp16 else 0) | (4 if fuse_silu else 0)
        flags |= sum(KERNEL_OPTIONS[name] for name in set(optimizations))
        cached = str(cache_dir.resolve()).encode() if cache_dir else None
        try:
            library_path = (bundle / "gaterv3.dll").resolve()
            if model_file is not None:
                # The Torch import above has already loaded cuBLASLt and cuDNN.
                self.dll = load_session_library(library_path)
                self.open_packed(model_file, batch, flags, cached)
            else:
                cuda, cudnn = libraries()
                for path in (cuda / "bin/x86_64", cudnn / "bin"):
                    self.dll_dirs.enter_context(os.add_dll_directory(str(path)))
                self.dll = ctypes.CDLL(str(library_path))
                self.dll.gt_error.restype = ctypes.c_char_p
                self.open_legacy(bundle, batch, graph, fp16, flags, cached)
            self.host_output = torch.empty((batch, *SHAPE), dtype=self.dtype)
        except BaseException:
            self.close()
            raise

    def open_legacy(self, bundle, batch, graph, fp16, flags, cached):
        create = self.dll.gt_create_fp16 if fp16 else self.dll.gt_create
        create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        create.restype = ctypes.c_void_p
        self.destroy = self.dll.gt_destroy
        self.destroy.argtypes = [ctypes.c_void_p]
        self.destroy.restype = None
        self.dll.gt_input.argtypes = [ctypes.c_void_p]
        self.dll.gt_input.restype = ctypes.c_uint64
        self.run = self.dll.gt_run_fp16 if fp16 else self.dll.gt_run
        self.run.argtypes = [ctypes.c_void_p] * 3
        self.dll.gt_enqueue.argtypes = [ctypes.c_void_p] * 2
        directory = str(bundle.resolve()).encode()
        if cached or flags & ~3:
            create_ex = self.dll.gt_create_ex
            create_ex.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_uint32,
                ctypes.c_char_p,
            ]
            create_ex.restype = ctypes.c_void_p
            self.handle = create_ex(directory, batch, flags, cached)
        else:
            self.handle = create(directory, batch, graph)
        if not self.handle:
            self.check(-1)
        self.device_input = self.dll.gt_input(self.handle)

    def open_packed(self, path, batch, flags, cached):
        self.destroy = self.dll.gt_session_close
        options = OptionsABI(ctypes.sizeof(OptionsABI), 0, flags, 0, cached)
        self.handle = self.dll.gt_session_open(str(path.resolve()).encode(), ctypes.byref(options))
        if not self.handle:
            self.check(-1)
        self.check(self.dll.gt_session_prepare(self.handle, batch, *SHAPE[1:]))

    def session_info(self):
        return read_session_info(self.dll, self.handle)

    def check(self, status):
        if status:
            raise RuntimeError(self.dll.gt_error().decode())

    def close(self):
        if self.handle:
            self.destroy(self.handle)
            self.handle = None
        self.dll_dirs.close()

    def host(self, x):
        if (
            x.device.type != "cpu"
            or x.dtype != torch.float32
            or not x.is_contiguous()
            or x.ndim != 4
            or x.shape[1] != 3
            or (self.model_file is None and x.shape != self.host_output.shape)
        ):
            raise ValueError("Native benchmark input must be contiguous FP32 NCHW at the runner's batch size")
        packed = x.to(dtype=self.dtype)
        if self.model_file is None:
            self.check(self.run(self.handle, packed.data_ptr(), self.host_output.data_ptr()))
        else:
            if x.shape != self.host_output.shape:
                self.host_output = torch.empty_like(packed)
            size = packed.nbytes
            self.check(
                self.dll.gt_session_run(
                    self.handle,
                    x.shape[0],
                    x.shape[2],
                    x.shape[3],
                    packed.data_ptr(),
                    size,
                    self.host_output.data_ptr(),
                    size,
                )
            )
        return self.host_output.float()

    def prepare_device(self, x):
        if self.model_file is not None:
            packed = x.to(dtype=self.dtype)
            output = torch.empty_like(packed)

            def run_packed():
                self.check(
                    self.dll.gt_session_enqueue(
                        self.handle,
                        x.shape[0],
                        x.shape[2],
                        x.shape[3],
                        packed.data_ptr(),
                        packed.nbytes,
                        output.data_ptr(),
                        output.nbytes,
                        torch.cuda.current_stream().cuda_stream,
                    )
                )

            return run_packed
        from cuda.bindings import driver

        packed = x.to(dtype=self.dtype)
        torch.cuda.synchronize()
        (status,) = driver.cuMemcpyDtoD(self.device_input, packed.data_ptr(), packed.numel() * packed.element_size())
        if status:
            raise RuntimeError(f"Native device input copy failed: {status}")

        def run():
            self.check(self.dll.gt_enqueue(self.handle, torch.cuda.current_stream().cuda_stream))

        return run

    def memory_values(self, symbol, keys):
        if self.model_file is not None:
            info = self.session_info()
            values = info.buffer_bytes if symbol == "gt_buffer_sizes" else info.startup_device_used_bytes
            return dict(zip(keys, values, strict=True))
        query = getattr(self.dll, symbol, None)
        if query is None:
            return None
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        query.restype = None
        sizes = (ctypes.c_uint64 * 4)()
        query(self.handle, sizes)
        return dict(zip(keys, sizes, strict=True))

    def setup(self):
        if self.model_file is not None:
            return session_setup(self.dll, self.handle, self.session_info())
        timing = getattr(self.dll, "gt_setup_timings", None)
        if timing is None:
            return None
        timing.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        timing.restype = None
        values = (ctypes.c_double * 4)()
        timing(self.handle, values)
        functions = (
            (self.dll.gt_cache_status, ctypes.c_int),
            (self.dll.gt_cache_message, ctypes.c_char_p),
            (self.dll.gt_fused_silu_count, ctypes.c_int),
            (self.dll.gt_fusion_message, ctypes.c_char_p),
        )
        for function, dtype in functions:
            function.argtypes = [ctypes.c_void_p]
            function.restype = dtype
        result = {
            "stages_ms": dict(
                zip(
                    ("model", "plans", "allocation_warmup", "capture"),
                    values,
                    strict=True,
                )
            ),
            "cache_status": CACHE_STATUSES[self.dll.gt_cache_status(self.handle)],
            "cache_message": self.dll.gt_cache_message(self.handle).decode(),
            "fused_silu_count": self.dll.gt_fused_silu_count(self.handle),
            "fusion_message": self.dll.gt_fusion_message(self.handle).decode(),
        }
        query = getattr(self.dll, "gt_optimization_stats", None)
        if query is not None:
            query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            query.restype = None
            counts = (ctypes.c_int * 5)()
            query(self.handle, counts)
            result["optimization"] = dict(
                zip(
                    OPTIMIZATION_STATS,
                    counts,
                    strict=True,
                )
            )
        return result

    def profile(self):
        if self.model_file is None:
            self.dll.gt_operation_count.argtypes = [ctypes.c_void_p]
            count = self.dll.gt_operation_count(self.handle)
            table = np.fromfile(self.bundle / "schedule.bin", dtype="<i4", offset=16)
            shapes = table[: 4 * (count + 1)].reshape(-1, 4)
            ops = table[4 * (count + 1) :].reshape(-1, 11)
            profile = self.dll.gt_profile
            profile.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
        else:
            specs, ops, _ = packed_schedule(self.model_file)
            count = len(ops)
            info = self.session_info()
            shapes = np.array([
                (1, 1, channels)
                if level < 0
                else (
                    info.padded_height >> level,
                    info.padded_width >> level,
                    channels,
                )
                for level, channels in specs
            ])
            profile = self.dll.gt_session_profile
        times = np.zeros(count, dtype=np.float32)
        self.check(profile(self.handle, times.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count))
        totals = defaultdict(float)
        rows = []
        for entry, ms in zip(ops, times, strict=True):
            kind, output, source = map(int, entry[:3])
            label = Op(kind).name.lower()
            if kind == Op.CONV:
                label = f"conv{int(entry[9])}x{int(entry[9])}"
            totals[label] += float(ms)
            rows.append({
                "operation": label,
                "input_shape": shapes[source, :3].tolist(),
                "output_shape": shapes[output, :3].tolist(),
                "milliseconds": float(ms),
            })
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
        options = {
            "precision": args.precision,
            "cache_dir": args.cache_dir,
            "fuse_silu": args.fuse_silu,
            "optimizations": enabled_optimizations(args),
            "model_file": args.model_file,
        }
        native = Native(args.bundle, 16, **options)
        runners.callback(native.close)
        single = Native(args.bundle, 1, **options)
        runners.callback(single.close)
        setup = {"16": native.setup(), "1": single.setup()}
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
                score = subprocess.check_output([str(METRIC), str(ref_file), str(native_file)], text=True)
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
    mse = sum(row["mse"] * count for row, count in zip(rows, pixels, strict=True)) / sum(pixels)
    return {
        "pages": rows,
        "aggregate_psnr_db": -10 * np.log10(mse) if mse else None,
        "ssimulacra2": {"mean": statistics.mean(scores), "minimum": min(scores)} if scores else None,
        "batch1": stress_rows,
        "setup": setup,
    }


def ort_session(onnx_path, profile_dir, cache_dir, batch):
    profiled, _ = create_rtx_session(onnx_path, batch, 128, cache_dir, profile_dir / onnx_path.stem)
    input_shape = profiled.get_inputs()[0].shape
    profiled.run(None, {"input": np.zeros(input_shape, dtype=np.float16)})
    events = json.loads(Path(profiled.end_profiling()).read_text(encoding="utf-8"))
    nodes = [event for event in events if event.get("cat") == "Node" and event.get("dur", 0)]
    placement = sorted({event.get("args", {}).get("provider", "") for event in nodes} - {""})
    if placement != ["NvTensorRTRTXExecutionProvider"]:
        raise RuntimeError(f"Unexpected RTX node placement {placement}")
    del profiled
    session, options = create_rtx_session(onnx_path, batch, 128, cache_dir)
    return session, placement, options


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
                    page[name].append({
                        "image": image_name,
                        "tile_index": index,
                        "top": top,
                        "left": left,
                        "seconds": seconds,
                    })
        for name, rows in page.items():
            reports[name]["per_tile"].extend(rows)
            reports[name]["per_image"].append({
                "image": image_name,
                "tiles": len(items),
                "source_pixels": image.shape[-2] * image.shape[-1],
                "average_tile_ms": statistics.mean(row["seconds"] for row in rows) * 1000,
            })
    return {
        name: {
            **row,
            "average_tile_ms": statistics.mean(sample["seconds"] for sample in row["per_tile"]) * 1000,
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
        rows.append({
            "image": name,
            "tiles": len(items),
            "source_pixels": pixels,
            "batch16_seconds": image_seconds,
            "batch16_megapixels_per_second": pixels / image_seconds / 1e6,
        })
    seconds = sum(row["batch16_seconds"] for row in rows)
    return {
        "per_image": rows,
        "batch16_seconds": seconds,
        "batch16_megapixels_per_second": sum(row["source_pixels"] for row in rows) / seconds / 1e6,
    }


def speed(args, images, batch, report, save):
    host_input = batch_input(patches(images[0][1], 128)[:batch], batch)
    device_input = host_input.cuda()
    device_half = device_input.half()
    torch.cuda.synchronize()
    with ExitStack() as stack:
        configured = enabled_optimizations(args)

        def native_case(
            bundle,
            precision,
            graph=True,
            control=False,
            fuse_silu=args.fuse_silu,
            optimizations=configured,
        ):
            instance = Native(
                bundle,
                batch,
                graph=graph,
                precision=precision,
                cache_dir=None if control else args.cache_dir,
                fuse_silu=fuse_silu and not control,
                optimizations=() if control else optimizations,
                model_file=None if control else args.model_file,
            )
            stack.callback(instance.close)
            return instance

        native = native_case(args.bundle, args.precision)
        cases = {"native_graph": (native.prepare_device(device_input), native.host)}
        report["setup"] = {"native_graph": native.setup()}
        if args.control_bundle:
            control = native_case(args.control_bundle, args.precision, control=True)
            cases["native_control"] = (
                control.prepare_device(device_input),
                control.host,
            )
        fp16 = None
        if args.compare_fp16:
            fp16 = native_case(args.bundle, "fp16")
            cases["native_fp16"] = (fp16.prepare_device(device_input), fp16.host)
            report["setup"]["native_fp16"] = fp16.setup()
        elif not args.control_bundle and not args.compare_silu and not args.compare_optimization:
            eager = native_case(args.bundle, args.precision, graph=False)
            cases["native_eager"] = (eager.prepare_device(device_input), eager.host)
        if args.compare_silu:
            for mode in ("mixed", "fp16") if args.compare_fp16 else (args.precision,):
                candidate = native_case(args.bundle, mode, fuse_silu=True)
                label = f"native_{mode}_silu"
                cases[label] = (candidate.prepare_device(device_input), candidate.host)
                report["setup"][label] = candidate.setup()
        candidates = {}
        for label, mode, optimizations in optimization_candidates(args):
            candidate = native_case(args.bundle, mode, optimizations=optimizations)
            cases[label] = (candidate.prepare_device(device_input), candidate.host)
            report["setup"][label] = candidate.setup()
            candidates[label] = candidate
        if "torch" in args.baselines:
            eager, _ = load_model(args.model, device="cuda", dtype=torch.float16)

            def torch_calls(model):
                return (
                    lambda: model(device_half),
                    lambda x: model(x.to(device="cuda", dtype=torch.float16)).float().cpu(),
                )

            cases["torch_eager"] = torch_calls(eager)
            if "inductor" in args.baselines:
                compiled = torch.compile(eager, mode="reduce-overhead", dynamic=False)
                cases["torch_inductor"] = torch_calls(compiled)
        if "rtx" in args.baselines:
            graph = args.baseline / f"gaterv3-fp16-t128-b{batch}.onnx"
            rtx_cache = args.rtx_cache_dir or args.baseline / "rtx-runtime-cache"
            session, placement, options = ort_session(graph, args.output, rtx_cache, batch)
            report["rtx_provider_placement"] = placement
            report["rtx_provider_options"] = options
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
                    "megapixels_per_second_at_median": batch * 128 * 128 / median / 1000,
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
            report["native_profile"] = native.profile()
            print(
                f"Batch {batch} native profile {report['native_profile']['totals_ms']}",
                flush=True,
            )
            save()
            if fp16 is not None:
                report["native_fp16_profile"] = fp16.profile()
                save()
            if candidates:
                report["optimization_profiles"] = {
                    label: candidate.profile() for label, candidate in candidates.items()
                }
                save()
        if not args.skip_pages:
            calls = {name: host for name, (_, host) in cases.items() if name != "native_eager"}
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
    configured = enabled_optimizations(args)
    runners = {"native_graph": (args.bundle, args.precision, args.fuse_silu, configured)}
    if args.control_bundle:
        runners["native_control"] = (args.control_bundle, args.precision, False, ())
    if args.compare_fp16:
        runners["native_fp16"] = (args.bundle, "fp16", args.fuse_silu, configured)
    if args.compare_silu:
        for mode in ("mixed", "fp16") if args.compare_fp16 else (args.precision,):
            runners[f"native_{mode}_silu"] = (args.bundle, mode, True, configured)
    for label, mode, optimizations in optimization_candidates(args):
        runners[label] = (args.bundle, mode, args.fuse_silu, optimizations)

    def used():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return total - free, total

    for label, (bundle, precision, fuse_silu, optimizations) in runners.items():
        torch.cuda.empty_cache()
        before, capacity = used()
        started = time.perf_counter()
        native = Native(
            bundle,
            batch,
            precision=precision,
            fuse_silu=fuse_silu,
            cache_dir=None if label == "native_control" else args.cache_dir,
            optimizations=optimizations,
            model_file=None if label == "native_control" else args.model_file,
        )
        initialization_ms = (time.perf_counter() - started) * 1000
        try:
            captured, _ = used()
            first_call = time.perf_counter()
            native.host(sample)
            time_to_first_output_ms = initialization_ms + (time.perf_counter() - first_call) * 1000
            for _ in range(7):
                native.host(sample)
            warmed, _ = used()
            buffers = native.memory_values("gt_buffer_sizes", ("weights", "activations", "workspace", "rgb_io"))
            startup = native.memory_values(
                "gt_startup_memory",
                ("context", "tuning_workspace", "tuned", "captured"),
            )
            setup = native.setup()
        finally:
            native.close()
        released, _ = used()
        row = {
            "initialization_wall_ms": initialization_ms,
            "time_to_first_output_ms": time_to_first_output_ms,
            "setup": setup,
            "device_capacity_bytes": capacity,
            "device_used_before_bytes": before,
            "device_used_captured_bytes": captured,
            "device_used_warmed_bytes": warmed,
            "device_used_released_bytes": released,
            "device_used_warmed_delta_bytes": warmed - before,
            "runner_buffers_bytes": buffers,
            "startup_device_used_bytes": startup,
            "startup_sampled_max_delta_bytes": max(captured, *startup.values()) - before if startup else None,
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
    parser.add_argument("--model-file", type=Path, help="Packed model for the native session API")
    parser.add_argument("--legacy", action="store_true", help="Use the fixed bundle API")
    parser.add_argument(
        "--control-bundle",
        type=Path,
        help="Fixed bundle control at the selected precision",
    )
    parser.add_argument("--cache-dir", type=Path, help="Read and write tuned plans in this directory")
    parser.add_argument("--fuse-silu", action="store_true", help="Try the cuDNN convolution/SiLU graph")
    parser.add_argument(
        "--compare-silu",
        action="store_true",
        help="Compare split and fused convolution/SiLU",
    )
    for option, description in (
        ("tune-projections", "Use a wider cuBLASLt search timed by CUDA Graph replay"),
        (
            "fuse-gate-pool",
            "Fuse the full-resolution local gate and spatial partial sums",
        ),
        (
            "fp16-qkv",
            "Use FP16 accumulation for the folded QKV projections in mixed precision",
        ),
    ):
        parser.add_argument(f"--{option}", action="store_true", help=description)
    parser.add_argument(
        "--compare-optimization",
        choices=tuple(KERNEL_OPTIONS),
        nargs="+",
        default=(),
        help="Benchmark each selected optimization independently against the current settings",
    )
    parser.add_argument("--precision", choices=("mixed", "fp16"), default="mixed")
    parser.add_argument(
        "--compare-fp16",
        action="store_true",
        help="Include full FP16 in the timing and memory comparisons",
    )
    parser.add_argument("--baseline", type=Path, default=ROOT / "output/baseline")
    parser.add_argument("--rtx-cache-dir", type=Path, help="TensorRT-RTX runtime cache directory")
    parser.add_argument("--output", type=Path, default=ROOT / "output/native-validation")
    parser.add_argument("--suite", choices=("quality", "speed", "memory", "all"), default="all")
    parser.add_argument(
        "--baselines",
        choices=("torch", "inductor", "rtx"),
        nargs="*",
        default=["torch", "inductor", "rtx"],
    )
    parser.add_argument("--batch", type=int, choices=(1, 16), nargs="+", default=[1, 16])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect an instrumented native op profile",
    )
    parser.add_argument("--skip-pages", action="store_true", help="Time repeated batches only")
    args = parser.parse_args()
    if args.legacy and args.model_file is not None:
        parser.error("--legacy and --model-file are mutually exclusive")
    if args.model_file is None and not args.legacy:
        model_file = args.bundle / "gaterv3.gtmodel"
        if model_file.is_file():
            args.model_file = model_file
    if args.rounds < 1 or args.repeats < 1:
        parser.error("Rounds and repeats must be positive")
    if "inductor" in args.baselines and "torch" not in args.baselines:
        parser.error("Include torch when benchmarking inductor")
    if args.compare_fp16 and args.precision != "mixed":
        parser.error("--compare-fp16 pairs the mixed runner with full FP16")
    if args.compare_silu and args.fuse_silu:
        parser.error("--compare-silu uses the split path as its control")
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
            name: digest(bundle / name) for name in ("gaterv3.dll", "kernels.cubin", "schedule.bin", "weights.bin")
        }
        if manifest["checkpoint_sha256"] != checkpoint_hash or any(
            hashes[name] != manifest["weights_and_schedule"][name] for name in ("schedule.bin", "weights.bin")
        ):
            parser.error(f"Native schedule or weights do not match the checkpoint manifest in {bundle}")
        return hashes

    if args.model_file is not None:
        _, _, packed_checkpoint = packed_schedule(args.model_file)
        if packed_checkpoint != checkpoint_hash:
            parser.error(f"Packed model does not match the checkpoint in {args.model}")
        bundle_hashes = {
            "gaterv3.dll": digest(args.bundle / "gaterv3.dll"),
            "model_file": digest(args.model_file),
        }
    else:
        bundle_hashes = checked_bundle(args.bundle)

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "checkpoint_sha256": checkpoint_hash,
        "bundle_sha256": bundle_hashes,
        "native_api": "packed_session" if args.model_file is not None else "legacy_fixed",
        "precision": args.precision,
        "fuse_silu": args.fuse_silu,
        "optimizations": enabled_optimizations(args),
    }
    if args.control_bundle:
        report["control_sha256"] = checked_bundle(args.control_bundle)

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

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
