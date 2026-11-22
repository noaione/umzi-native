"""Build the Umzi session library and bundle it in platform wheels.

Hatchling runs ``NativeLibraryHook`` for every standard wheel, including the
wheel pip builds while installing from source. ``scripts/build_native.py``
runs the same ``build()`` from the command line. NVRTC compiles the CUDA
kernels into cubins and PTX, which are embedded in one host library:
``umzi_native.dll`` built with MSVC on Windows, or ``libumzi_native.so`` built with
the system C++ compiler on Linux. The CLI is linked from the same object.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / "native"
KERNEL_SOURCES = {family: NATIVE / "kernels" / family for family in ("gaterv3", "gaterv2", "moesr")}
EXACT_MATH_FAMILIES = frozenset({"gaterv2", "moesr"})
HOST_SOURCES = (
    *(source / "runner.cpp" for source in KERNEL_SOURCES.values()),
    NATIVE / "runtime" / "api.cpp",
    NATIVE / "runtime" / "registry.cpp",
    NATIVE / "runtime" / "model_package.cpp",
    NATIVE / "runtime" / "plan_cache_file.cpp",
)
WINDOWS = sys.platform == "win32"
LIBRARY_NAME = "umzi_native.dll" if WINDOWS else "libumzi_native.so"
CLI_NAME = "umzi_native.exe" if WINDOWS else "umzi_native"
CUBIN_TARGETS = ("sm_86", "sm_89", "sm_120")
PTX_TARGET = "compute_86"
NVRTC = "bin/x86_64/nvrtc64_130_0.dll" if WINDOWS else "lib/libnvrtc.so.13"
CUDNN = "bin/cudnn64_9.dll" if WINDOWS else "lib/libcudnn.so.9"
# The pip wheels that supply headers, NVRTC, import libraries and link targets.
# cuBLASLt and cuDNN are only linked; the wheel does not bundle them.
BUILD_REQUIREMENTS = (
    "nvidia-cuda-runtime>=13.4,<13.5",
    "nvidia-cuda-nvrtc>=13.0.88,<13.1",
    "nvidia-cuda-crt>=13.4,<13.5",
    "nvidia-cuda-cccl>=13.3,<14",
    "nvidia-cublas>=13.7,<14",
    "nvidia-cudnn-cu13>=9.26,<10",
)


# NamedTuple rather than dataclass: Hatchling executes this file without
# registering it in sys.modules, which dataclasses require.
class Toolchain(NamedTuple):
    cuda: Path
    nvrtc: Path
    cudnn: Path
    includes: tuple[Path, ...]


class BuildSettings(NamedTuple):
    """Optional overrides; every path defaults to the installed NVIDIA wheels."""

    arches: Sequence[str] = CUBIN_TARGETS
    cuda_root: Path | None = None
    nvrtc_root: Path | None = None
    cudnn_root: Path | None = None
    crt_include: Path | None = None
    cccl_include: Path | None = None
    openssl_include: Sequence[Path] = ()
    cuda_driver_lib: Path | None = None
    test_kernel: bool = False


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _nvidia_dirs() -> list[Path]:
    # Search sys.path, which covers the isolated build environment pip creates,
    # as well as the interpreter's site-packages directories.
    import site

    roots = [*sys.path, *site.getsitepackages()]
    return [Path(path) / "nvidia" for path in dict.fromkeys(roots) if path]


def _find(supplied: Path | None, relative: str, candidates: Iterable[Path], what: str) -> Path:
    if supplied is not None:
        if not (supplied / relative).exists():
            raise FileNotFoundError(f"{what}: {supplied / relative} does not exist")
        return supplied.resolve()
    for candidate in candidates:
        if (candidate / relative).exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{what} ({relative}) was not found. Install the CUDA 13 build wheels: {', '.join(BUILD_REQUIREMENTS)}"
    )


def find_toolchain(settings: BuildSettings) -> Toolchain:
    sites = _nvidia_dirs()
    cuda = _find(settings.cuda_root, "include/cuda.h", (site / "cu13" for site in sites), "CUDA 13 headers")
    nvrtc = _find(settings.nvrtc_root, NVRTC, (cuda,), "CUDA 13.0 NVRTC") / NVRTC
    cudnn = _find(settings.cudnn_root, CUDNN, (site / "cudnn" for site in sites), "cuDNN 9")
    header = cuda / "include" / "cuda.h"
    if not header.is_file() or not re.search(
        r"^#define\s+CUDA_VERSION\s+13\d{3}\b", header.read_text(encoding="utf-8"), re.MULTILINE
    ):
        raise RuntimeError(f"Expected CUDA 13 headers at {header}")
    _find(None, "include/cudnn.h", (cudnn,), "cuDNN headers")
    headers = cuda / "include"
    crt = _find(settings.crt_include, "crt/mma.h", (headers,), "CUDA CRT headers")
    cccl = _find(settings.cccl_include, "nv/target", (headers, headers / "cccl"), "CUDA CCCL headers")
    return Toolchain(cuda, nvrtc, cudnn, tuple(dict.fromkeys((headers, crt, cccl))))


def _load_nvrtc(path: Path) -> ctypes.CDLL:
    if WINDOWS:
        # NVRTC loads its version-matched builtins DLL by name through the
        # standard search order, which add_dll_directory does not affect.
        os.environ["PATH"] = os.pathsep.join((str(path.parent), os.environ.get("PATH", "")))
    nvrtc = ctypes.CDLL(str(path))
    major, minor = ctypes.c_int(), ctypes.c_int()
    if nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) or (major.value, minor.value) != (13, 0):
        raise RuntimeError(
            f"Expected CUDA 13.0 NVRTC at {path} to generate PTX for the CUDA 13 driver baseline. "
            "Use pip wheel . or point UMZI_NVRTC_ROOT at the NVIDIA CUDA 13.0 NVRTC wheel."
        )
    _log(f"CUDA NVRTC {major.value}.{minor.value}")
    return nvrtc


def _compile_kernels(
    nvrtc: ctypes.CDLL, includes: Sequence[Path], arch: str, source: Path, *, fast_math: bool = True
) -> bytes:
    _log(f"Compiling {source.name} kernels for {arch}")
    prog = ctypes.c_void_p()

    def check(status: int) -> None:
        if status:
            raise RuntimeError(f"NVRTC status {status} for {arch}")

    code = (source / "kernels.cu").read_bytes()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), code, b"kernels.cu", 0, None, None))
    try:
        options = [
            f"--gpu-architecture={arch}".encode(),
            *[f"--include-path={path}".encode() for path in (*includes, source)],
            b"--std=c++17",
        ]
        if fast_math:
            options.append(b"--use_fast_math")
        argv = (ctypes.c_char_p * len(options))(*options)
        status = nvrtc.nvrtcCompileProgram(prog, len(options), argv)
        size = ctypes.c_size_t()
        check(nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size)))
        if size.value > 1:
            log = ctypes.create_string_buffer(size.value)
            check(nvrtc.nvrtcGetProgramLog(prog, log))
            _log(log.value.decode(errors="replace").rstrip())
        check(status)
        kind = "CUBIN" if arch.startswith("sm_") else "PTX"
        check(getattr(nvrtc, f"nvrtcGet{kind}Size")(prog, ctypes.byref(size)))
        image = ctypes.create_string_buffer(size.value)
        check(getattr(nvrtc, f"nvrtcGet{kind}")(prog, image))
        return image.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))


def _embed_images(output: Path, cubins: dict[str, bytes], ptx: bytes, family: str) -> None:
    hex_bytes = [f"0x{value:02x}" for value in range(256)]
    with (output / f"{family}_images.h").open("w", encoding="ascii") as header:
        header.write(f'#pragma once\n#include "cuda_module.h"\nnamespace {family}_images {{\n')
        for name, data in (*cubins.items(), (PTX_TARGET, ptx)):
            header.write(f"inline constexpr unsigned char {name}[] = {{\n")
            for start in range(0, len(data), 24):
                header.write(", ".join(hex_bytes[byte] for byte in data[start : start + 24]))
                header.write(",\n")
            header.write("};\n")
        header.write("inline constexpr umzi::KernelImage embedded_cubins[] = {\n")
        for name in cubins:
            header.write(f"    {{{name[3:]}, {name}, sizeof({name})}},\n")
        header.write(
            f"}};\ninline constexpr umzi::KernelImage embedded_ptx = "
            f"{{{PTX_TARGET[8:]}, {PTX_TARGET}, sizeof({PTX_TARGET})}};\n}}\n"
        )


def _msvc_environment() -> dict[str, str]:
    if os.environ.get("INCLUDE") and os.environ.get("LIB") and shutil.which("cl"):
        return dict(os.environ)
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        raise RuntimeError("Visual Studio with the MSVC x64 tools is required to build umzi_native.dll")
    install = subprocess.check_output(
        [
            str(vswhere),
            *("-latest", "-products", "*"),
            *("-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"),
            *("-property", "installationPath"),
        ],
        text=True,
    ).strip()
    if not install:
        raise RuntimeError("Visual Studio with the MSVC x64 tools is required to build umzi_native.dll")
    setup = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
    output = subprocess.check_output(f'cmd.exe /d /s /c ""{setup}" >nul && set"', text=True)
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line and not line.startswith("="))


def _run(*args: str, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    # Keep stdout clean for build frontends; tool output goes to stderr.
    subprocess.run(args, env=env, cwd=cwd, check=True, stdout=sys.stderr)


def _build_windows(output: Path, toolchain: Toolchain, includes: Sequence[Path], test_kernel: bool) -> Path:
    env = _msvc_environment()
    # CreateProcess searches the parent PATH for a bare executable name.
    # Resolve cl.exe through the MSVC environment and use its companion tools.
    msvc_path = next((value for name, value in env.items() if name.casefold() == "path"), "")
    compiler = shutil.which("cl.exe", path=msvc_path)
    if compiler is None:
        raise RuntimeError("cl.exe was not found in the MSVC environment")
    cl = Path(compiler).resolve()
    librarian, linker = cl.with_name("lib.exe"), cl.with_name("link.exe")
    for tool in (librarian, linker):
        if not tool.is_file():
            raise FileNotFoundError(f"MSVC tool not found: {tool}")

    # Keep the compiler and linker scratch files in the build directory.
    # Some environments let Python write to %TEMP% but deny it to link.exe.
    scratch = output / "tmp"
    scratch.mkdir(exist_ok=True)
    env = {**env, "TEMP": str(scratch), "TMP": str(scratch)}
    objects = []
    flags = ("/nologo", "/std:c++17", "/O2", "/EHsc", "/MD", "/wd4996")
    if test_kernel:
        flags += ("/DUMZI_TEST_KERNEL",)
    for cpp in HOST_SOURCES:
        obj = output / "_".join(cpp.relative_to(NATIVE).with_suffix(".obj").parts)
        _run(str(cl), *flags, "/c", *(f"/I{path}" for path in includes), f"/Fo{obj}", str(cpp), env=env, cwd=output)
        objects.append(str(obj))
    # The development wheels omit the cuBLASLt and cuDNN MSVC import
    # libraries, so generate them from the functions the runtime calls.
    headers = (path for source in KERNEL_SOURCES.values() for path in sorted(source.glob("*.h")))
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (*HOST_SOURCES, *headers))
    imports = []
    for dll, symbol in (("cudnn64_9", r"cudnn\w+"), ("cublasLt64_13", r"cublasLt\w+")):
        names = set(re.findall(rf"\b({symbol})\(", sources))
        if dll.startswith("cublas"):
            names.add("cublasLtMatmulPreferenceDestroy")
        definition = output / f"{dll}.def"
        exports = "\n".join(sorted(names))
        definition.write_text(f"LIBRARY {dll}.dll\nEXPORTS\n{exports}\n", encoding="ascii")
        library = output / f"{dll}.lib"
        _run(str(librarian), "/nologo", "/machine:x64", f"/def:{definition}", f"/out:{library}", env=env, cwd=output)
        imports.append(str(library))
    link = (str(linker), "/nologo", *objects, str(toolchain.cuda / "lib/x64/cuda.lib"), "bcrypt.lib", "version.lib")
    library = output / LIBRARY_NAME
    _run(*link, *imports, "/DLL", f"/OUT:{library}", env=env, cwd=output)
    _run(*link, *imports, f"/OUT:{output / CLI_NAME}", f"/IMPLIB:{output / 'umzi_native-cli.lib'}", env=env, cwd=output)
    return library


def _build_linux(
    output: Path, toolchain: Toolchain, includes: Sequence[Path], driver: Path | None, test_kernel: bool
) -> Path:
    compiler = os.environ.get("CXX", "c++")
    objects = []
    extra_flags = ("-DUMZI_TEST_KERNEL",) if test_kernel else ()
    for cpp in HOST_SOURCES:
        obj = output / "_".join(cpp.relative_to(NATIVE).with_suffix(".o").parts)
        _run(
            compiler,
            *("-std=c++17", "-O3", "-fPIC", "-fvisibility=hidden", "-Wno-deprecated-declarations", "-c"),
            *extra_flags,
            *(f"-I{path}" for path in includes),
            *("-o", str(obj), str(cpp)),
        )
        objects.append(str(obj))
    if driver is None:
        # The driver is linked by soname; any copy of libcuda.so.1 will do.
        driver = next(
            (path for path in (Path("/usr/lib/wsl/lib/libcuda.so.1"),) if path.is_file()),
            None,
        )
    link = [
        compiler,
        *objects,
        "-Wl,-z,defs",
        "-Wl,-rpath,$ORIGIN",
        str(driver) if driver else "-l:libcuda.so.1",
        f"-L{toolchain.cuda / 'lib'}",
        f"-L{toolchain.cudnn / 'lib'}",
        "-l:libcublasLt.so.13",
        "-l:libcudnn.so.9",
        "-l:libcrypto.so.3",
        "-ldl",
    ]
    library = output / LIBRARY_NAME
    _run(*link, "-shared", "-o", str(library))
    _run(*link, "-o", str(output / CLI_NAME))
    return library


def build(output: Path, settings: BuildSettings | None = None) -> Path:
    """Compile the kernels, link the session library and CLI, and return the library path."""
    if not WINDOWS and sys.platform != "linux":
        raise RuntimeError("GateRV3 builds support Windows and Linux")
    settings = settings or BuildSettings()
    unknown = set(settings.arches) - set(CUBIN_TARGETS)
    if unknown or not settings.arches:
        raise ValueError(f"Cubin targets must be chosen from {', '.join(CUBIN_TARGETS)}")
    toolchain = find_toolchain(settings)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nvrtc = _load_nvrtc(toolchain.nvrtc)
    for family, source in KERNEL_SOURCES.items():
        fast_math = family not in EXACT_MATH_FAMILIES
        cubins = {
            arch: _compile_kernels(nvrtc, toolchain.includes, arch, source, fast_math=fast_math)
            for arch in dict.fromkeys(settings.arches)
        }
        ptx = _compile_kernels(nvrtc, toolchain.includes, PTX_TARGET, source, fast_math=fast_math)
        for arch, image in cubins.items():
            (output / f"{family}-{arch}.cubin").write_bytes(image)
        (output / f"{family}-{PTX_TARGET}.ptx").write_bytes(ptx)
        if family == "gaterv3":
            # The legacy fixed-bundle runner takes a cubin from its bundle directory.
            (output / "kernels.cubin").write_bytes(next(iter(cubins.values())))
        _embed_images(output, cubins, ptx, family)
    includes = (*toolchain.includes, toolchain.cudnn / "include", NATIVE / "common", output)
    if WINDOWS:
        library = _build_windows(output, toolchain, includes, settings.test_kernel)
    else:
        library = _build_linux(
            output, toolchain, (*includes, *settings.openssl_include), settings.cuda_driver_lib, settings.test_kernel
        )
    _log(f"Built {library}")
    return library


def _paths(value: str | None) -> list[Path]:
    return [Path(item) for item in (value or "").split(os.pathsep) if item]


def settings_from_environment() -> BuildSettings:
    """Read build overrides from ``UMZI_*`` environment variables.

    Build frontends pass no arguments to the hook, so the same overrides that
    ``scripts/build_native.py`` takes as flags are read from the environment.
    Path lists use the platform path separator.
    """
    env = os.environ.get

    def path(name: str) -> Path | None:
        value = env(name)
        return Path(value) if value else None

    arches = [arch for arch in (env("UMZI_CUDA_ARCHES") or "").replace(",", " ").split() if arch]
    return BuildSettings(
        arches=arches or CUBIN_TARGETS,
        cuda_root=path("UMZI_CUDA_ROOT"),
        nvrtc_root=path("UMZI_NVRTC_ROOT"),
        cudnn_root=path("UMZI_CUDNN_ROOT"),
        crt_include=path("UMZI_CRT_INCLUDE"),
        cccl_include=path("UMZI_CCCL_INCLUDE"),
        openssl_include=_paths(env("UMZI_OPENSSL_INCLUDE")),
        cuda_driver_lib=path("UMZI_CUDA_DRIVER_LIB"),
    )


try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # Imported by scripts/build_native.py without Hatchling.
    pass
else:

    class NativeLibraryHook(BuildHookInterface):
        """Compile the native library for each standard wheel and bundle it.

        ``UMZI_NATIVE_LIBRARY`` names an already built library to bundle
        instead, for builds whose toolchain runs outside the wheel build.
        ``UMZI_NATIVE_BUILD_DIR`` keeps the build products, including the
        CLI linked from the same object, in that directory.
        """

        def _build_dir(self) -> tuple[Path, bool]:
            kept = os.environ.get("UMZI_NATIVE_BUILD_DIR")
            if kept:
                return Path(kept).resolve(), True
            return Path(self.directory) / "umzi-native-build", False

        PLUGIN_NAME = "custom"

        def dependencies(self) -> list[str]:
            # Hatchling cannot call this for a script-defined hook, so the
            # same requirements are listed in [build-system] requires.
            return list(BUILD_REQUIREMENTS)

        def initialize(self, version: str, build_data: dict) -> None:
            if self.target_name != "wheel":
                return
            if version == "editable":
                # Editable installs import the sources; build the library
                # with scripts/build_native.py and pass SessionOptions(dll_path=...).
                return
            prebuilt = os.environ.get("UMZI_NATIVE_LIBRARY")
            if prebuilt:
                library = Path(prebuilt).resolve()
                if library.name != LIBRARY_NAME or not library.is_file():
                    raise FileNotFoundError(f"UMZI_NATIVE_LIBRARY must name an existing {LIBRARY_NAME}: {library}")
            else:
                output, kept = self._build_dir()
                if output.exists() and not kept:
                    shutil.rmtree(output)
                library = build(output, settings_from_environment())
            build_data["force_include"][str(library)] = f"umzi_native/{LIBRARY_NAME}"
            # The library is loaded through ctypes, so the wheel depends on the
            # operating system and machine but not on the Python version or ABI.
            platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
            build_data["pure_python"] = False
            build_data["tag"] = f"py3-none-{platform}"

        def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
            output, kept = self._build_dir()
            if not kept:
                shutil.rmtree(output, ignore_errors=True)
