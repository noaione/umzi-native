"""Export the original GateRV3 checkpoint as a native model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

import torch
from reference.model import load_model
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]

MODEL_MAGIC = b"GTR3PCK\0"
MODEL_FORMAT_VERSION = 1
MODEL_HEADER = struct.Struct("<8sII32s")
MODEL_SECTION = struct.Struct("<IIQQ")
MODEL_METADATA = struct.Struct("<8I")


class Op(IntEnum):
    PACK = 0
    CONV = 1
    RMS = 2
    LOCAL = 3
    MEAN_PROJECT = 4
    LOCAL_SKIP = 5
    INCEPTION = 6
    GLOBAL_SKIP = 7
    SHUFFLE = 8
    CONCAT = 9
    SILU = 10
    SPAN = 11
    DEPTHWISE = 12
    ATTENTION = 13
    GATE = 14
    ADD = 15
    RGB = 16
    SHUFFLE_CONCAT = 17


class Activation(NamedTuple):
    index: int
    first: int
    last: int
    size: int

    def overlaps(self, other: Activation) -> bool:
        return self.first <= other.last and other.first <= self.last


def pack_activations(activations: list[Activation]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    placed: list[Activation] = []
    for tensor in sorted(activations, key=lambda item: (-item.size, item.first)):
        offset = 0
        conflicts = sorted(
            (offsets[other.index], other.size)
            for other in placed
            if tensor.overlaps(other)
        )
        for start, size in conflicts:
            if offset + tensor.size <= start:
                break
            offset = max(offset, start + size)
        offsets[tensor.index] = offset
        placed.append(tensor)

    for i, tensor in enumerate(activations):
        start = offsets[tensor.index]
        if start % 256:
            raise RuntimeError(f"Misaligned arena offset for tensor {tensor.index}")
        for other in activations[:i]:
            addr = offsets[other.index]
            if tensor.overlaps(other) and max(start, addr) < min(
                start + tensor.size, addr + other.size
            ):
                raise RuntimeError(
                    f"Arena overlap for tensors {tensor.index} and {other.index}"
                )
    return offsets


class Schedule:
    def __init__(self, weights):
        self.weights = weights
        self.shapes = [(128, 128, 3)]
        self.ops: list[tuple[int, ...]] = []
        self.last_use = [-1]

    def weight(self, value: Tensor | None, *, dtype=torch.float16) -> int:
        if value is None:
            return -1
        offset = (self.weights.tell() + 255) & ~255
        self.weights.write(bytes(offset - self.weights.tell()))
        self.weights.write(value.detach().to(dtype).contiguous().numpy().tobytes())
        return offset

    def op(self, kind, shape, *inputs, weights=(), args=(0, 0)):
        output = len(self.shapes)
        self.shapes.append(shape)
        for index in inputs:
            self.last_use[index] = len(self.ops)
        self.last_use.append(len(self.ops))
        self.ops.append(
            (
                int(kind),
                output,
                *inputs,
                *([-1] * (4 - len(inputs))),
                *weights,
                *([-1] * (3 - len(weights))),
                *args,
            )
        )
        return output

    def conv(self, a, layer: nn.Conv2d, *, rgb_in=False, rgb_out=False):
        h, w, c = self.shapes[a]
        k = layer.kernel_size[0]
        assert layer.groups == 1 and layer.in_channels == (3 if rgb_in else c)
        assert layer.kernel_size == (k, k) and layer.padding == (k // 2, k // 2)
        assert layer.stride == layer.dilation == (1, 1)
        weight, bias = layer.weight, layer.bias
        if rgb_in:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, 5))
        if rgb_out:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, 0, 0, 5))
            if bias is not None:
                bias = torch.nn.functional.pad(bias, (0, 5))
        return self.op(
            Op.CONV,
            (h, w, weight.shape[0]),
            a,
            weights=(self.weight(weight.permute(0, 2, 3, 1)), self.weight(bias)),
            args=(k, 0),
        )

    def norm(self, a, layer):
        assert layer.eps == 1e-6
        previous = self.ops[a - 1]
        # The fused launch reads the residual inputs while producing both
        # logical outputs. Keep them live so the norm output cannot alias them.
        residual_inputs = (
            tuple(i for i in previous[2:6] if i >= 0)
            if previous[0] in (Op.LOCAL_SKIP, Op.GLOBAL_SKIP)
            else ()
        )
        return self.op(
            Op.RMS,
            self.shapes[a],
            a,
            *residual_inputs,
            weights=(self.weight(layer.scale), self.weight(layer.offset)),
        )

    def local(self, a, block):
        h, w, c = self.shapes[a]
        norm, project, spatial, _ = block.local
        projected = self.conv(self.norm(a, norm), project)
        assert spatial.groups == c and spatial.weight.shape == (2 * c, 2, 3, 3)
        gated = self.op(
            Op.LOCAL,
            (h, w, c),
            projected,
            weights=(
                self.weight(spatial.weight.permute(2, 3, 1, 0)),
                self.weight(spatial.bias),
            ),
        )
        layer = block.sca[1]
        scale = self.op(
            Op.MEAN_PROJECT,
            (1, 1, c),
            gated,
            weights=(self.weight(layer.weight), self.weight(layer.bias)),
        )
        return self.op(
            Op.LOCAL_SKIP,
            (h, w, c),
            gated,
            scale,
            a,
            weights=(self.weight(block.gamma0),),
        )

    def global_block(self, a, block, residual=None):
        h, w, c = self.shapes[a]
        normalized = self.norm(a, block.norm)
        mixed = block.token_mix
        assert block.split_indices == [3 * c // 2, c // 2, c]
        if hasattr(mixed, "qkv"):
            assert (h, w, c, mixed.num_heads, mixed.flash_enabled) == (
                8,
                8,
                512,
                16,
                True,
            )
            # The final C channels of fc1 feed only the linear QKV projection.
            # Compose those weights and biases before packing the result.
            first = block.fc1.weight.flatten(1)
            assert block.fc1.bias is not None and mixed.qkv.bias is None
            last = mixed.qkv.weight.flatten(1)
            weight = torch.cat((first[: 2 * c], last @ first[2 * c :]))
            bias = torch.cat((block.fc1.bias[: 2 * c], last @ block.fc1.bias[2 * c :]))
            projected = self.op(
                Op.CONV,
                (h, w, 5 * c),
                normalized,
                weights=(self.weight(weight), self.weight(bias)),
                args=(1, 0),
            )
            spatial = mixed.qkv_dwconv
            assert spatial.groups == 3 * c and spatial.weight.shape == (3 * c, 1, 3, 3)
            qkv = self.op(
                Op.DEPTHWISE,
                (h, w, 3 * c),
                projected,
                weights=(
                    self.weight(spatial.weight.permute(2, 3, 1, 0)),
                    self.weight(spatial.bias),
                ),
            )
            attended = self.op(Op.ATTENTION, (h, w, c), qkv)
            attended = self.conv(attended, mixed.project_out)
            gated = self.op(Op.GATE, (h, w, 3 * c // 2), projected, attended)
        else:
            projected = self.conv(normalized, block.fc1)
            assert mixed.split_indexes == [5 * c // 8, c // 8, c // 8, c // 8]
            layers = (mixed.dwconv_hw, mixed.dwconv_w, mixed.dwconv_h)
            weight = torch.cat(
                [layer.weight.flatten(1).T.flatten() for layer in layers]
            )
            bias = torch.cat([layer.bias for layer in layers])
            gated = self.op(
                Op.INCEPTION,
                (h, w, 3 * c // 2),
                projected,
                weights=(self.weight(weight), self.weight(bias)),
            )
        out = self.conv(gated, block.fc2)
        inputs, weights = (
            ((out,), ()) if residual is None else ((out, a), (self.weight(residual),))
        )
        return self.op(Op.GLOBAL_SKIP, (h, w, c), *inputs, weights=weights)

    def meta(self, a, blocks):
        for block in blocks:
            a = self.global_block(self.local(a, block), block.glob, block.gamma1)
        return a

    def span(self, a, block):
        first = self.conv(a, block.c1_r.eval_conv)
        h, w, c = self.shapes[first]
        first = self.op(Op.SILU, self.shapes[first], first)
        current = self.conv(first, block.c2_r.eval_conv)
        current = self.op(Op.SILU, self.shapes[current], current)
        current = self.conv(current, block.c3_r.eval_conv)
        return self.op(Op.SPAN, (h, w, c), current, a), first

    def concat(self, *inputs):
        h, w, _ = self.shapes[inputs[0]]
        return self.op(
            Op.CONCAT, (h, w, sum(self.shapes[i][2] for i in inputs)), *inputs
        )

    def build(self, model):
        image = self.op(Op.PACK, (128, 128, 8), 0)
        features = self.conv(image, model.in_to_dim, rgb_in=True)
        span, _ = self.span(features, model.span_block0)
        shortcut = span
        for block in model.span_n_b:
            span, _ = self.span(span, block)
        span, first = self.span(span, model.span_end)
        span = self.conv(span, model.sisr_end_conv.eval_conv)
        span = self.conv(
            self.concat(features, span, shortcut, first), model.sisr_cat_conv
        )

        current = features
        shortcuts = []
        for block in model.gater_encode:
            current = self.meta(current, block.gated)
            shortcuts.append(current)
            current = self.conv(current, block.scale[0])
            h, w, c = self.shapes[current]
            current = self.op(Op.SHUFFLE, (h // 2, w // 2, 4 * c), current, args=(0, 0))
        for block in model.latent:
            current = self.global_block(current, block)
        for block, shortcut in zip(model.decode, reversed(shortcuts), strict=True):
            current = self.conv(current, block.scale[0])
            h, w, c = self.shapes[current]
            assert self.shapes[shortcut] == (h * 2, w * 2, c // 4)
            merged = self.op(
                Op.SHUFFLE_CONCAT, (h * 2, w * 2, c // 2), current, shortcut
            )
            current = self.conv(merged, block.shor)
            current = self.meta(current, block.gated)
        current = self.op(Op.ADD, self.shapes[span], current, span)
        current = self.conv(current, model.dim_to_in, rgb_out=True)
        self.op(
            Op.RGB,
            (128, 128, 3),
            current,
            0,
            weights=(
                self.weight(model.gamma, dtype=torch.float32),
                self.weight(model.gamma),
            ),
        )

    def write(self, path: Path, weight_bytes: int):
        last_use = self.last_use.copy()
        for index, (conv, silu) in enumerate(zip(self.ops, self.ops[1:])):
            if conv[0] == Op.CONV and conv[9] == 3 and silu[0] == Op.SILU:
                # An optional fused launch reads the original convolution input
                # while writing the SiLU result.
                source = conv[2]
                last_use[source] = max(last_use[source], index + 1)
        activations = []
        for index, (h, w, c) in enumerate(self.shapes[1:-1], start=1):
            first = index - 1
            op = self.ops[first]
            # Residual + RMSNorm emits both outputs in the residual launch.
            if op[0] == Op.RMS and op[3] >= 0:
                first -= 1
            size = (h * w * c * 2 + 255) & ~255
            activations.append(Activation(index, first, last_use[index], size))
        offsets = pack_activations(activations)
        with path.open("wb") as file:
            file.write(
                struct.pack(
                    "<4sIII",
                    b"GTR3",
                    len(self.shapes),
                    len(self.ops),
                    weight_bytes,
                )
            )
            for index, (h, w, c) in enumerate(self.shapes):
                file.write(struct.pack("<4i", h, w, c, offsets.get(index, -1)))
            for op in self.ops:
                file.write(struct.pack("<11i", *op))
        return max(offsets[tensor.index] + tensor.size for tensor in activations)

    def write_model(self, path: Path, weights: Path, checkpoint_sha256: str):
        metadata = MODEL_METADATA.pack(
            3, 1, 16, 16, 32, 3, len(self.shapes), len(self.ops)
        )
        levels = {128 >> level: level for level in range(5)}
        levels[1] = -1
        schedule = b"".join(
            struct.pack("<2i", levels[h], c) for h, _w, c in self.shapes
        ) + b"".join(struct.pack("<11i", *op) for op in self.ops)
        identity = bytes.fromhex(checkpoint_sha256)
        sizes = (len(metadata), len(schedule), weights.stat().st_size, len(identity))
        offset = MODEL_HEADER.size + len(sizes) * MODEL_SECTION.size
        sections = []
        for index, size in enumerate(sizes, start=1):
            sections.append(MODEL_SECTION.pack(index, 0, offset, size))
            offset += size
        with path.open("w+b") as file:
            file.seek(MODEL_HEADER.size)
            file.write(b"".join(sections))
            file.write(metadata)
            file.write(schedule)
            with weights.open("rb") as source:
                shutil.copyfileobj(source, file)
            file.write(identity)
            file.seek(MODEL_HEADER.size)
            digest = hashlib.file_digest(file, "sha256").digest()
            file.seek(0)
            file.write(
                MODEL_HEADER.pack(MODEL_MAGIC, MODEL_FORMAT_VERSION, len(sizes), digest)
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/1x_umzi_digital_decompress_gaterv3_1.pth",
    )
    parser.add_argument(
        "--output", type=Path, help="Model file, or bundle directory with --legacy"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Export the fixed schedule and weights separately",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, config = load_model(args.model, device="cpu")
    actual = (
        config.in_ch,
        config.dim,
        config.enc_blocks,
        config.dec_blocks,
        config.num_latent,
        config.sisr_blocks,
        config.scale,
        config.attention,
        config.flash,
    )
    if actual != (3, 32, (2, 2, 4, 8), (2, 2, 2, 2), 12, 4, 1, True, True):
        raise ValueError(
            f"Unsupported native checkpoint configuration {config.describe()}"
        )
    default_dir = ROOT / "output/native"
    # Existing callers that pass a bundle directory keep the original export.
    legacy = args.legacy or (args.output is not None and args.output.suffix == "")
    output = args.output or (default_dir if legacy else default_dir / "gaterv3.gtmodel")
    parent = output if legacy else output.parent
    parent.mkdir(parents=True, exist_ok=True)
    with args.model.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    with tempfile.TemporaryDirectory(dir=parent) as staging:
        directory = Path(staging)
        weights_path = directory / "weights.bin"
        with torch.inference_mode(), weights_path.open("wb") as weights:
            schedule = Schedule(weights)
            schedule.build(model)
        if not legacy:
            model_path = directory / "model"
            schedule.write_model(model_path, weights_path, digest)
            model_path.replace(output)
            print(
                f"Exported {output} with {len(schedule.ops)} operations\nCheckpoint {digest}"
            )
            return
        arena = schedule.write(directory / "schedule.bin", weights_path.stat().st_size)
        artifacts = ("schedule.bin", "weights.bin")
        hashes = {}
        for name in artifacts:
            with (directory / name).open("rb") as file:
                hashes[name] = hashlib.file_digest(file, "sha256").hexdigest()
        (directory / "checkpoint.json").write_text(
            json.dumps(
                {"checkpoint_sha256": digest, "weights_and_schedule": hashes}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        for name in (*artifacts, "checkpoint.json"):
            (directory / name).replace(output / name)
    print(
        f"Checkpoint {digest}, {len(schedule.ops)} operations, {arena:,} arena bytes per tile"
    )


if __name__ == "__main__":
    main()
