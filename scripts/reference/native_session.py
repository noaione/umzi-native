"""Device-buffer validation adapter for packaged sessions."""

from __future__ import annotations

import ctypes

import torch
from torch import Tensor

from umzi_native import InferenceSession
from umzi_native._abi import IoABI, ShapeABI, check_status


class DeviceSession:
    """Enqueue CUDA tensors on the current Torch stream through the session ABI."""

    def __init__(self, session: InferenceSession):
        self.session = session
        self.dtype = torch.float16 if session.get_inputs()[0].type == "tensor(float16)" else torch.float32

    def __call__(self, sample: Tensor) -> Tensor:
        if sample.dtype != self.dtype or not sample.is_cuda or not sample.is_contiguous():
            raise ValueError("The session expects a contiguous CUDA tensor of its I/O dtype")
        dims = ShapeABI.from_dimensions(sample.shape)
        io = IoABI(ctypes.sizeof(IoABI))
        dll, handle = self.session._dll, self.session._handle
        check_status(dll, dll.umzi_session_describe(handle, ctypes.byref(dims), ctypes.byref(io)))
        result = torch.empty(io.output.shape.dimensions_tuple(), dtype=self.dtype, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        check_status(
            dll,
            dll.umzi_session_enqueue(
                handle,
                ctypes.byref(dims),
                sample.data_ptr(),
                io.input.bytes,
                result.data_ptr(),
                io.output.bytes,
                stream,
            ),
        )
        return result
