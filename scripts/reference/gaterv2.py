"""GateRV2 attention with FP32 normalization and contractions."""

from __future__ import annotations

import torch
from resselt.archs.gaterv2.arch import Attention, GateRV2
from torch import Tensor, nn


def normalize_channels(value: Tensor) -> Tensor:
    inverse_norm = 1 / torch.linalg.vector_norm(value, dim=1)
    return torch.einsum("bcn,bn->bcn", value, inverse_norm)


def linear_attention(query: Tensor, key: Tensor, value: Tensor, eps: float) -> Tensor:
    query = normalize_channels(query).transpose(1, 2)
    key = normalize_channels(key)
    denominator = query.shape[1] + torch.einsum("bnc,bc->bn", query, key.sum(dim=-1) + eps)
    value_sum = torch.einsum("bcn->bc", value).unsqueeze(-1)
    matrix = torch.einsum("bmn,bcn->bmc", key, value)
    numerator = value_sum + torch.einsum("bnm,bmc->bcn", query, matrix)
    return torch.einsum("bcn,bn->bcn", numerator, 1 / denominator)


class FP32Attention(nn.Module):
    def __init__(self, layer: Attention):
        super().__init__()
        self.query_conv = layer.query_conv
        self.key_conv = layer.key_conv
        self.value_conv = layer.value_conv
        self.eps = layer.eps

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        query = self.query_conv(x).reshape(batch, -1, height * width).float()
        key = self.key_conv(x).reshape(batch, -1, height * width).float()
        value = self.value_conv(x).reshape(batch, -1, height * width).float()
        return linear_attention(query, key, value, self.eps).reshape(batch, channels, height, width).to(x.dtype)


def use_fp32_attention(model: nn.Module) -> None:
    if not isinstance(model, GateRV2):
        raise TypeError("FP32 latent attention requires a GateRV2 checkpoint")
    for block in model.latent:
        if not isinstance(block.token_mix, Attention):
            raise TypeError("GateRV2 latent block is missing linear attention")
        block.token_mix = FP32Attention(block.token_mix).train(block.token_mix.training)
