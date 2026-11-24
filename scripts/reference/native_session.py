"""Device-buffer validation adapter for packaged sessions."""

from __future__ import annotations

import ctypes

import torch
from torch import Tensor

from umzi_native import InferenceSession
from umzi_native._abi import IoABI, ShapeABI, check_status


class DeviceSession:
    """Enqueue CUDA tensors on the current Torch stream through the session ABI.

    With reuse_output, consume each result before the next call overwrites it.
    """

    def __init__(self, session: InferenceSession, *, reuse_output: bool = False):
        self.session = session
        self.dtype = torch.float16 if session.get_inputs()[0].type == "tensor(float16)" else torch.float32
        self.reuse_output = reuse_output
        self.bound: tuple[torch.Size, ShapeABI, IoABI, Tensor] | None = None

    def __call__(self, sample: Tensor) -> Tensor:
        if sample.dtype != self.dtype or not sample.is_cuda or not sample.is_contiguous():
            raise ValueError("The session expects a contiguous CUDA tensor of its I/O dtype")
        dll, handle = self.session._dll, self.session._handle
        if self.bound is not None and self.bound[0] == sample.shape and self.bound[3].device == sample.device:
            _, dims, io, result = self.bound
        else:
            dims = ShapeABI.from_dimensions(sample.shape)
            io = IoABI(ctypes.sizeof(IoABI))
            check_status(dll, dll.umzi_session_describe(handle, ctypes.byref(dims), ctypes.byref(io)))
            result = torch.empty(io.output.shape.dimensions_tuple(), dtype=self.dtype, device=sample.device)
            if self.reuse_output:
                self.bound = (sample.shape, dims, io, result)
        stream = torch.cuda.current_stream(sample.device).cuda_stream
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
