"""Device enumeration for the separate Vulkan runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path

from ._abi import check_status
from ._loader import PathArg, find_session_library, load_session_library


class _DeviceABI(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "struct_size",
            "api_version",
            "driver_version",
            "vendor_id",
            "device_id",
            "device_type",
            "subgroup_size",
            "shader_float16",
            "storage_buffer16",
            "cooperative_matrix",
            "max_storage_buffer_range",
            "max_workgroup_invocations",
            "max_shared_memory",
        )
    ] + [
        ("max_workgroup_count", ctypes.c_uint32 * 3),
        ("device_name", ctypes.c_char * 256),
        ("driver_name", ctypes.c_char * 256),
        ("driver_info", ctypes.c_char * 256),
    ]


def vulkan_devices(dll_path: PathArg | None = None) -> list[dict]:
    """Return Vulkan GPU capabilities in session device-index order."""
    library = find_session_library(Path.cwd() / "model.ummodel", dll_path, backend="vulkan")
    dll = load_session_library(library, backend="vulkan")
    dll.umzi_vulkan_device_count.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    dll.umzi_vulkan_device_count.restype = ctypes.c_int
    dll.umzi_vulkan_device.argtypes = [ctypes.c_uint32, ctypes.POINTER(_DeviceABI)]
    dll.umzi_vulkan_device.restype = ctypes.c_int
    count = ctypes.c_uint32()
    check_status(dll, dll.umzi_vulkan_device_count(ctypes.byref(count)))
    result = []
    for index in range(count.value):
        device = _DeviceABI(ctypes.sizeof(_DeviceABI))
        check_status(dll, dll.umzi_vulkan_device(index, ctypes.byref(device)))
        version = device.api_version
        result.append({
            "index": index,
            "device_name": device.device_name.decode(),
            "driver_name": device.driver_name.decode(),
            "driver_info": device.driver_info.decode(),
            "api_version": f"{(version >> 22) & 0x7F}.{(version >> 12) & 0x3FF}.{version & 0xFFF}",
            "driver_version": device.driver_version,
            "vendor_id": device.vendor_id,
            "device_id": device.device_id,
            "device_type": device.device_type,
            "subgroup_size": device.subgroup_size,
            "shader_float16": bool(device.shader_float16),
            "storage_buffer16": bool(device.storage_buffer16),
            "cooperative_matrix_extension": bool(device.cooperative_matrix),
            "max_storage_buffer_range": device.max_storage_buffer_range,
            "max_workgroup_invocations": device.max_workgroup_invocations,
            "max_shared_memory": device.max_shared_memory,
            "max_workgroup_count": tuple(device.max_workgroup_count),
        })
    return result
