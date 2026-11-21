"""Build the native GateRV3 runtime with the CUDA 13 development wheels."""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "gaterv3"
WINDOWS = sys.platform == "win32"
CUBIN_TARGETS = ("sm_86", "sm_89", "sm_120")
PTX_TARGET = "compute_86"
NVRTC = "bin/x86_64/nvrtc64_130_0.dll" if WINDOWS else "lib/libnvrtc.so.13"
CUDNN = "bin/cudnn64_9.dll" if WINDOWS else "lib/libcudnn.so.9"


def libraries(cuda_root=None, cudnn_root=None):
    sites = [Path(path) / "nvidia" for path in (*sys.path, *site.getsitepackages())]
    cuda = cuda_root or next((site / "cu13" for site in sites if (site / "cu13" / NVRTC).is_file()), None)
    cudnn = cudnn_root or next((site / "cudnn" for site in sites if (site / "cudnn" / CUDNN).is_file()), None)
    if cuda is None or cudnn is None:
        raise RuntimeError('Install the CUDA 13 build tools with pip install ".[dev]"')
    header = cuda / "include" / "cuda.h"
    if not header.is_file() or not re.search(
        r"^#define\s+CUDA_VERSION\s+13\d{3}\b", header.read_text(encoding="utf-8"), re.M
    ):
        raise RuntimeError(f"Expected CUDA 13 headers at {header}")
    for path in (cuda / NVRTC, cudnn / CUDNN, cudnn / "include/cudnn.h"):
        if not path.is_file():
            raise FileNotFoundError(path)
    return cuda.resolve(), cudnn.resolve()


def compiler_environment():
    if os.environ.get("INCLUDE") and os.environ.get("LIB") and shutil.which("cl"):
        return os.environ
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    install = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    if not install:
        raise RuntimeError("MSVC x64 tools were not found")
    setup = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
    output = subprocess.check_output(
        f'cmd.exe /d /s /c ""{setup}" >nul && set"',
        text=True,
    )
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line and not line.startswith("="))


def load_nvrtc(cuda):
    path = cuda / NVRTC
    if WINDOWS:
        # NVRTC also uses PATH to resolve its version-matched builtins.
        os.environ["PATH"] = str(path.parent) + os.pathsep + os.environ["PATH"]
    nvrtc = ctypes.CDLL(str(path))
    major, minor = ctypes.c_int(), ctypes.c_int()
    if nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) or major.value != 13:
        raise RuntimeError(f"Expected CUDA 13 NVRTC at {path}")
    print(f"CUDA NVRTC {major.value}.{minor.value}", flush=True)
    return nvrtc


def compile_kernels(nvrtc, includes, arch):
    print(f"Compiling GateRV3 for {arch}", flush=True)
    prog = ctypes.c_void_p()

    def check(status):
        if status:
            raise RuntimeError(f"NVRTC status {status} for {arch}")

    code = (SOURCE / "kernels.cu").read_bytes()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), code, b"kernels.cu", 0, None, None))
    try:
        options = [
            f"--gpu-architecture={arch}".encode(),
            *[f"--include-path={path}".encode() for path in (*includes, SOURCE)],
            b"--std=c++17",
            b"--use_fast_math",
        ]
        argv = (ctypes.c_char_p * len(options))(*options)
        status = nvrtc.nvrtcCompileProgram(prog, len(options), argv)
        size = ctypes.c_size_t()
        check(nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size)))
        if size.value > 1:
            log = ctypes.create_string_buffer(size.value)
            check(nvrtc.nvrtcGetProgramLog(prog, log))
            print(log.value.decode(errors="replace"), end="")
        check(status)
        if arch.startswith("sm_"):
            check(nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)))
            image = ctypes.create_string_buffer(size.value)
            check(nvrtc.nvrtcGetCUBIN(prog, image))
        else:
            check(nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size)))
            image = ctypes.create_string_buffer(size.value)
            check(nvrtc.nvrtcGetPTX(prog, image))
        return image.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))


def embed_images(output, cubins, ptx):
    with (output / "cuda_images.h").open("w", encoding="ascii") as header:
        header.write(
            "#pragma once\n#include <cstddef>\nnamespace gt {\n"
            "struct KernelImage { int capability; const unsigned char* data; std::size_t size; };\n"
        )
        hex_bytes = [f"0x{value:02x}" for value in range(256)]
        for name, data in (*cubins.items(), (PTX_TARGET, ptx)):
            header.write(f"inline constexpr unsigned char {name}[] = {{\n")
            for start in range(0, len(data), 24):
                header.write(", ".join(hex_bytes[byte] for byte in data[start : start + 24]))
                header.write(",\n")
            header.write("};\n")
        header.write("inline constexpr KernelImage embedded_cubins[] = {\n")
        for name in cubins:
            header.write(f"    {{{name[3:]}, {name}, sizeof({name})}},\n")
        header.write(
            f"}};\ninline constexpr KernelImage embedded_ptx = "
            f"{{{PTX_TARGET[8:]}, {PTX_TARGET}, sizeof({PTX_TARGET})}};\n}}\n"
        )


def build_windows(output, cuda, includes):
    env = compiler_environment()
    cpp = SOURCE / "runner.cpp"
    obj = output / "runner.obj"

    def run(*args):
        subprocess.run(args, env=env, cwd=output, check=True)

    run(
        "cl.exe",
        "/nologo",
        "/std:c++17",
        "/O2",
        "/EHsc",
        "/MD",
        "/c",
        "/wd4996",
        *(f"/I{path}" for path in includes),
        f"/Fo{obj}",
        str(cpp),
    )
    # The development wheels omit the cuBLASLt and cuDNN MSVC import libraries.
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (cpp, *sorted(SOURCE.glob("*.h"))))
    imports = []
    for dll, symbol in (("cudnn64_9", r"cudnn\w+"), ("cublasLt64_13", r"cublasLt\w+")):
        names = set(re.findall(rf"\b({symbol})\(", sources))
        if dll.startswith("cublas"):
            names.add("cublasLtMatmulPreferenceDestroy")
        definition = output / f"{dll}.def"
        definition.write_text(
            f"LIBRARY {dll}.dll\nEXPORTS\n" + "\n".join(sorted(names)) + "\n",
            encoding="ascii",
        )
        library = output / f"{dll}.lib"
        run(
            "lib.exe",
            "/nologo",
            "/machine:x64",
            f"/def:{definition}",
            f"/out:{library}",
        )
        imports.append(str(library))
    link = (
        "link.exe",
        "/nologo",
        str(obj),
        str(cuda / "lib/x64/cuda.lib"),
        "bcrypt.lib",
        "version.lib",
        *imports,
    )
    library = output / "gaterv3.dll"
    run(*link, "/DLL", f"/OUT:{library}")
    run(*link, f"/OUT:{output / 'gaterv3.exe'}", f"/IMPLIB:{output / 'gaterv3-cli.lib'}")
    return library


def build_linux(output, cuda, cudnn, includes, driver):
    cpp = SOURCE / "runner.cpp"
    obj = output / "runner.o"
    compiler = os.environ.get("CXX", "c++")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-fvisibility=hidden",
            "-Wno-deprecated-declarations",
            "-c",
            *(f"-I{path}" for path in includes),
            "-o",
            str(obj),
            str(cpp),
        ],
        check=True,
    )
    if driver is None:
        driver = next(
            (
                path
                for path in (
                    Path("/usr/lib/wsl/lib/libcuda.so.1"),
                    cuda / "lib/stubs/libcuda.so",
                )
                if path.is_file()
            ),
            None,
        )
    link = [
        compiler,
        str(obj),
        "-Wl,-z,defs",
        "-Wl,-rpath,$ORIGIN",
        str(driver) if driver else "-l:libcuda.so.1",
        f"-L{cuda / 'lib'}",
        f"-L{cudnn / 'lib'}",
        "-l:libcublasLt.so.13",
        "-l:libcudnn.so.9",
        "-l:libcrypto.so.3",
        "-ldl",
    ]
    library = output / "libgaterv3.so"
    subprocess.run([*link, "-shared", "-o", str(library)], check=True)
    subprocess.run([*link, "-o", str(output / "gaterv3")], check=True)
    return library


def bundle_wheel(library, directory):
    with tempfile.TemporaryDirectory(prefix="umzi-native-wheel-") as temp:
        staging = Path(temp)
        for filename in ("pyproject.toml", "hatch_build.py"):
            shutil.copy2(ROOT / filename, staging)
        package = staging / "src" / "umzi_native"
        shutil.copytree(
            ROOT / "src" / "umzi_native",
            package,
            # Package only the library built by this run.
            ignore=shutil.ignore_patterns("__pycache__", "*.dll", "*.so"),
        )
        shutil.copy2(library, package)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(directory),
            ],
            cwd=staging,
            check=True,
        )


def find_include(supplied, relative, *candidates):
    path = supplied or next(
        (candidate for candidate in candidates if (candidate / relative).is_file()),
        None,
    )
    if path is None or not (path / relative).is_file():
        raise RuntimeError(f'Missing {relative}. Install the CUDA 13 tools with pip install ".[dev]"')
    return path.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/native")
    parser.add_argument(
        "--arch",
        choices=CUBIN_TARGETS,
        action="append",
        help="Override the embedded cubin targets. Repeat to include multiple targets.",
    )
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--cudnn-root", type=Path)
    parser.add_argument("--crt-include", type=Path)
    parser.add_argument("--cccl-include", type=Path)
    parser.add_argument("--openssl-include", type=Path, action="append", default=[])
    parser.add_argument("--cuda-driver-lib", type=Path)
    parser.add_argument(
        "--wheel",
        action="store_true",
        help="Bundle the native library in a Python platform wheel",
    )
    args = parser.parse_args()
    if not WINDOWS and sys.platform != "linux":
        parser.error("GateRV3 builds support Windows and Linux")
    cuda, cudnn = libraries(args.cuda_root, args.cudnn_root)
    headers = cuda / "include"
    toolchain = ROOT / "output" / "native-toolchain"
    crt = find_include(args.crt_include, "crt/mma.h", headers, toolchain / "nvidia/cu13/include")
    cccl = find_include(
        args.cccl_include,
        "nv/target",
        headers,
        headers / "cccl",
        toolchain / "cccl/nvidia/cu13/include",
    )
    includes = tuple(dict.fromkeys((headers, crt, cccl)))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nvrtc = load_nvrtc(cuda)
    cubins = {arch: compile_kernels(nvrtc, includes, arch) for arch in dict.fromkeys(args.arch or CUBIN_TARGETS)}
    ptx = compile_kernels(nvrtc, includes, PTX_TARGET)
    for arch, image in cubins.items():
        (output / f"kernels-{arch}.cubin").write_bytes(image)
    (output / "kernels.cubin").write_bytes(next(iter(cubins.values())))
    (output / f"kernels-{PTX_TARGET}.ptx").write_bytes(ptx)
    embed_images(output, cubins, ptx)
    host_includes = (*includes, cudnn / "include", output)
    if WINDOWS:
        library = build_windows(output, cuda, host_includes)
    else:
        library = build_linux(
            output,
            cuda,
            cudnn,
            (*host_includes, *args.openssl_include),
            args.cuda_driver_lib,
        )
    print(f"Built {library}", flush=True)
    if args.wheel:
        bundle_wheel(library, output / "wheels")


if __name__ == "__main__":
    main()
