"""Tag wheels that bundle a native GateRV3 library for the build platform."""

from __future__ import annotations

import sysconfig
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

LIBRARIES = ("gaterv3.dll", "libgaterv3.so")


class PlatformWheelHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return
        package = Path(self.root) / "src" / "umzi_native"
        if not any((package / name).is_file() for name in LIBRARIES):
            # Without a library the wheel stays pure Python (py3-none-any).
            return
        # The library is loaded through ctypes, so it does not depend on the
        # Python version or ABI, only on the operating system and machine.
        platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform}"
