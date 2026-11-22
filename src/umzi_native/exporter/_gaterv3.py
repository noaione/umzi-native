"""Build the GateRV3 operation schedule and packed weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, cast

import torch
from resselt.archs.gaterv3.arch import Block, GateRV3
from torch import Tensor, nn

from .._model import ARCHITECTURE, GATERV3_FAMILY, OPERATION, SCHEMAS, TENSOR_SPEC
from .._model import GateRV3Op as Op
from ._loader import attention_flags
from ._package import write_package

SCHEMA_VERSION = SCHEMAS["gaterv3"]


class Schedule:
    def __init__(self, weights: BinaryIO):
        self.weights = weights
        self.shapes = [(128, 128, 3)]
        self.ops: list[tuple[int, ...]] = []

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
        self.ops.append((
            int(kind),
            output,
            *inputs,
            *([-1] * (4 - len(inputs))),
            *weights,
            *([-1] * (3 - len(weights))),
            *args,
        ))
        return output

    def conv(self, a, layer: nn.Conv2d, *, rgb_in=False, rgb_out=False):
        h, w, c = self.shapes[a]
        k = layer.kernel_size[0]
        if (
            layer.groups != 1
            or layer.in_channels != (3 if rgb_in else c)
            or layer.kernel_size != (k, k)
            or layer.padding != (k // 2, k // 2)
            or layer.stride != (1, 1)
            or layer.dilation != (1, 1)
        ):
            raise ValueError(f"Unsupported GateRV3 convolution: {layer}")
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
        # The packed kernel hard-codes this epsilon; reject any other value.
        if layer.eps != 1e-6:  # ruff: ignore[float-equality-comparison]
            raise ValueError(f"Unsupported GateRV3 RMSNorm epsilon: {layer.eps}")
        previous = self.ops[a - 1]
        # The fused launch reads the residual inputs while producing both
        # logical outputs. Keep them live so the norm output cannot alias them.
        residual_inputs = (
            tuple(i for i in previous[2:6] if i >= 0) if previous[0] in (Op.LOCAL_SKIP, Op.GLOBAL_SKIP) else ()
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
        if spatial.groups != c or spatial.weight.shape != (2 * c, 2, 3, 3):
            raise ValueError(f"Unsupported GateRV3 local convolution: {spatial}")
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
        if tuple(block.split_indices) != (3 * c // 2, c // 2, c):
            raise ValueError(f"Unsupported GateRV3 global split widths: {block.split_indices}")
        if hasattr(mixed, "qkv"):
            if (h, w, c, mixed.num_heads) != (8, 8, 512, 16):
                raise ValueError("GateRV3 spatial attention requires 16 heads and a 512-channel latent stage")
            # The final C channels of fc1 feed only the linear QKV projection.
            # Compose those weights and biases before packing the result.
            first = block.fc1.weight.flatten(1)
            if block.fc1.bias is None or mixed.qkv.bias is not None:
                raise ValueError("GateRV3 spatial attention requires a biased input projection and bias-free QKV")
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
            if spatial.groups != 3 * c or spatial.weight.shape != (3 * c, 1, 3, 3):
                raise ValueError(f"Unsupported GateRV3 QKV depthwise convolution: {spatial}")
            qkv = self.op(
                Op.DEPTHWISE,
                (h, w, 3 * c),
                projected,
                weights=(
                    self.weight(spatial.weight.permute(2, 3, 1, 0)),
                    self.weight(spatial.bias),
                ),
            )
            attended = self.op(
                Op.ATTENTION, (h, w, c), qkv, weights=(self.weight(mixed.temperature, dtype=torch.float32),)
            )
            attended = self.conv(attended, mixed.project_out)
            gated = self.op(Op.GATE, (h, w, 3 * c // 2), projected, attended)
        else:
            projected = self.conv(normalized, block.fc1)
            if tuple(mixed.split_indexes) != (5 * c // 8, c // 8, c // 8, c // 8):
                raise ValueError(f"Unsupported GateRV3 Inception split widths: {mixed.split_indexes}")
            layers = (mixed.dwconv_hw, mixed.dwconv_w, mixed.dwconv_h)
            weight = torch.cat([layer.weight.flatten(1).T.flatten() for layer in layers])
            bias = torch.cat([layer.bias for layer in layers])
            gated = self.op(
                Op.INCEPTION,
                (h, w, 3 * c // 2),
                projected,
                weights=(self.weight(weight), self.weight(bias)),
            )
        out = self.conv(gated, block.fc2)
        inputs, weights = ((out,), ()) if residual is None else ((out, a), (self.weight(residual),))
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
        return self.op(Op.CONCAT, (h, w, sum(self.shapes[i][2] for i in inputs)), *inputs)

    def build(self, model):
        image = self.op(Op.PACK, (128, 128, 8), 0)
        features = self.conv(image, model.in_to_dim, rgb_in=True)
        span, _ = self.span(features, model.span_block0)
        shortcut = span
        for block in model.span_n_b:
            span, _ = self.span(span, block)
        span, first = self.span(span, model.span_end)
        span = self.conv(span, model.sisr_end_conv.eval_conv)
        span = self.conv(self.concat(features, span, shortcut, first), model.sisr_cat_conv)

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
            if self.shapes[shortcut] != (h * 2, w * 2, c // 4):
                raise ValueError("Incompatible GateRV3 encoder and decoder feature shapes")
            merged = self.op(Op.SHUFFLE_CONCAT, (h * 2, w * 2, c // 2), current, shortcut)
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

    def write_model(self, path: Path, weights: Path, checkpoint_sha256: bytes, alignment: int) -> None:
        metadata = ARCHITECTURE.pack(3, 1, alignment, alignment, 32, 3, len(self.shapes), len(self.ops))
        levels = {128 >> level: level for level in range(5)}
        levels[1] = -1
        schedule = b"".join(TENSOR_SPEC.pack(levels[h], c) for h, _w, c in self.shapes) + b"".join(
            OPERATION.pack(*op) for op in self.ops
        )
        write_package(
            path, GATERV3_FAMILY, SCHEMA_VERSION, {1: metadata, 2: schedule, 3: weights, 4: checkpoint_sha256}
        )


def build_schedule(model: nn.Module, state: Mapping[str, Tensor], weights: Path) -> tuple[Schedule, int]:
    if not isinstance(model, GateRV3):
        raise ValueError("The GateRV3 packer requires a Resselt GateRV3 model")
    enc = tuple(len(cast("Block", block).gated) for block in model.gater_encode)
    dec = tuple(len(cast("Block", block).gated) for block in model.decode)
    attention = tuple(hasattr(block.token_mix, "qkv") for block in model.latent)
    if (
        model.scale != 1
        or model.in_to_dim.in_channels != 3
        or model.in_to_dim.out_channels != 32
        or len(model.span_n_b) != 4
    ):
        raise ValueError("GateRV3 export requires scale 1, RGB input, base width 32, and four SPAN blocks")
    topology = (enc, dec, attention)
    if topology not in (
        ((2, 2, 4, 8), (2, 2, 2, 2), (True,) * 12),
        ((2, 2, 4), (2, 2, 2), (False,) * 8),
    ):
        raise ValueError(f"Unsupported GateRV3 topology: encoder={enc}, decoder={dec}, latent attention={attention}")
    if any(attention):
        for name, flag in attention_flags(state).items():
            if flag.numel() != 1 or flag.item() != 1:
                raise ValueError(f"GateRV3 export requires spatial attention. Incompatible checkpoint flag {name!r}")

    with torch.inference_mode(), weights.open("wb") as file:
        model.to(device="cpu", dtype=torch.float32).eval()
        schedule = Schedule(file)
        schedule.build(model)
    return schedule, model.pad
