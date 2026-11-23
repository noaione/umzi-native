"""Sample device-wide GPU memory independently of the inference backend."""

from __future__ import annotations

import ctypes
import os
import statistics
import threading
import time
from typing import Any

MIB = 2**20


class _MemoryInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


class Nvml:
    def __init__(self, uuid: str | None = None) -> None:
        self.library = ctypes.CDLL("C:/Windows/System32/nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.library.nvmlDeviceGetHandleByUUID.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.library.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemoryInfo)]
        self._check(self.library.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        try:
            if uuid is None:
                status = self.library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle))
            else:
                status = self.library.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.handle))
            self._check(status)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status: int) -> None:
        if status:
            raise RuntimeError(f"NVML memory query failed ({status})")

    def read(self) -> dict[str, float]:
        info = _MemoryInfo()
        self._check(self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)))
        return {"used_mib": info.used / MIB, "total_mib": info.total / MIB}

    def close(self) -> None:
        self._check(self.library.nvmlShutdown())


class NvmlSampler:
    def __init__(self, nvml: Nvml, interval: float = 0.005) -> None:
        self.nvml = nvml
        self.interval = interval
        self.samples: list[list[float]] = []
        self.done = threading.Event()
        self.error: Exception | None = None
        self.start = 0.0
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        self.samples.append([time.perf_counter() - self.start, self.nvml.read()["used_mib"]])

    def _loop(self) -> None:
        try:
            while not self.done.wait(self.interval):
                self._sample()
        except Exception as error:
            self.error = error

    def __enter__(self) -> NvmlSampler:
        self.start = time.perf_counter()
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.done.set()
        if self.thread is not None:
            self.thread.join()
        self._sample()
        if self.error is not None:
            raise self.error

    def report(self, background: float) -> dict[str, Any]:
        values = [value for _, value in self.samples]
        return {
            "device_peak_mib": max(values),
            "device_median_mib": statistics.median(values),
            "peak_above_pre_context_background_mib": max(values) - background,
            "samples": len(values),
            "requested_sampling_interval_ms": self.interval * 1000,
        }
