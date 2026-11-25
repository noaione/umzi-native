"""Check the dynamic GateRV3 channel-attention cubin against Torch CUDA FP32."""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import torch
from torch.nn import functional


def check(status: int) -> None:
    if status:
        raise RuntimeError(f"CUDA driver returned {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cubin", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(91)
    torch.empty(1, device="cuda:0")
    driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    pointer = ctypes.c_void_p
    driver.cuModuleLoadData.argtypes = [ctypes.POINTER(pointer), pointer]
    driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(pointer), pointer, ctypes.c_char_p]
    driver.cuModuleUnload.argtypes = [pointer]
    driver.cuLaunchKernel.argtypes = [pointer, *([ctypes.c_uint] * 7), pointer, ctypes.POINTER(pointer), pointer]

    def launch(function: ctypes.c_void_p, blocks: int, *parameters: ctypes.c_void_p | ctypes.c_int) -> None:
        entries = (pointer * len(parameters))(*(ctypes.addressof(value) for value in parameters))
        stream = torch.cuda.current_stream().cuda_stream
        check(driver.cuLaunchKernel(function, blocks, 1, 1, 256, 1, 1, 0, stream, entries, None))

    module = pointer()
    image = ctypes.create_string_buffer(args.cubin.read_bytes())
    check(driver.cuModuleLoadData(ctypes.byref(module), image))
    try:
        score, apply = pointer(), pointer()
        check(driver.cuModuleGetFunction(ctypes.byref(score), module, b"channel_attention_probabilities"))
        check(driver.cuModuleGetFunction(ctypes.byref(apply), module, b"channel_attention_apply"))
        for batch, pixels, zero_qk in (
            (2, 63, True),
            (3, 75, False),
            (2, 256, False),
            (1, 1024, False),
            (1, 4096, False),
        ):
            qkv = torch.randn(batch, pixels, 1536, device="cuda:0", dtype=torch.float16)
            if zero_qk:
                qkv[..., :1024].zero_()
            temperature = torch.linspace(1, 4, 16, device="cuda:0", dtype=torch.float32)
            probs = torch.empty((batch, 16, 32, 32), device="cuda:0", dtype=torch.float16)
            out = torch.empty((batch, pixels, 512), device="cuda:0", dtype=torch.float16)
            launch(
                score,
                batch * 16,
                pointer(probs.data_ptr()),
                pointer(qkv.data_ptr()),
                pointer(temperature.data_ptr()),
                ctypes.c_int(pixels),
            )
            launch(
                apply,
                batch * 16 * ((pixels + 63) // 64),
                pointer(out.data_ptr()),
                pointer(qkv.data_ptr()),
                pointer(probs.data_ptr()),
                ctypes.c_int(pixels),
            )
            q, k, v = (
                tensor.reshape(batch, pixels, 16, 32).permute(0, 2, 3, 1) for tensor in qkv.float().chunk(3, dim=-1)
            )
            expected_probs = (
                (functional.normalize(q, dim=-1) @ functional.normalize(k, dim=-1).mT)
                * temperature[None, :, None, None]
            ).softmax(dim=-1)
            expected = (expected_probs.half().float() @ v).permute(0, 3, 1, 2).reshape_as(out).half()
            torch.testing.assert_close(probs, expected_probs.half(), atol=1e-4, rtol=1e-3)
            torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
            max_error = (out.float() - expected.float()).abs().max().item()
            print(f"batch {batch}, {pixels} positions, max output error {max_error:.6f}")
    finally:
        torch.cuda.synchronize()
        check(driver.cuModuleUnload(module))


if __name__ == "__main__":
    main()
