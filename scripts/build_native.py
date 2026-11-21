"""Build the Windows GateRV3 runner using MSVC and the system Python CUDA wheels."""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import site
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "native"


def libraries():
    for path in map(Path, site.getsitepackages()):
        nvidia = path / "nvidia"
        cuda, cudnn = nvidia / "cu13", nvidia / "cudnn"
        required = (
            cuda / "include" / "cuda.h",
            cuda / "bin" / "x86_64" / "nvrtc64_130_0.dll",
            cuda / "lib" / "x64" / "cuda.lib",
            cudnn / "include" / "cudnn.h",
            cudnn / "bin" / "cudnn64_9.dll",
        )
        if all(path.is_file() for path in required):
            return cuda, cudnn
    raise RuntimeError(
        "Could not find the NVIDIA cu13 and cuDNN wheels in system Python"
    )


def compiler_environment():
    if os.environ.get("INCLUDE") and os.environ.get("LIB") and shutil.which("cl"):
        return os.environ
    vswhere = (
        Path(os.environ["ProgramFiles(x86)"])
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
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
    # Running the batch file and querying set in the same process imports its environment.
    output = subprocess.check_output(
        f'cmd.exe /d /s /c ""{setup}" >nul && set"',
        text=True,
    )
    return dict(
        line.split("=", 1)
        for line in output.splitlines()
        if "=" in line and not line.startswith("=")
    )


def compile_kernels(cuda, crt_include, destination, arch):
    bin_dir = cuda / "bin" / "x86_64"
    # NVRTC resolves its builtins with the CUDA DLL loader's PATH search.
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
    with os.add_dll_directory(str(bin_dir)):
        nvrtc = ctypes.CDLL(str(bin_dir / "nvrtc64_130_0.dll"))
        prog = ctypes.c_void_p()

        def check(status):
            if status:
                raise RuntimeError(f"NVRTC status {status}")

        code = (SOURCE / "kernels.cu").read_bytes()
        check(
            nvrtc.nvrtcCreateProgram(
                ctypes.byref(prog), code, b"kernels.cu", 0, None, None
            )
        )
        try:
            options = [
                f"--gpu-architecture={arch}".encode(),
                f"--include-path={cuda / 'include'}".encode(),
                f"--include-path={crt_include}".encode(),
                f"--include-path={SOURCE}".encode(),
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
            check(nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)))
            cubin = ctypes.create_string_buffer(size.value)
            check(nvrtc.nvrtcGetCUBIN(prog, cubin))
            destination.write_bytes(cubin.raw)
        finally:
            check(nvrtc.nvrtcDestroyProgram(ctypes.byref(prog)))


def build(output: Path, arch: str, crt_include: Path | None, cccl_include: Path | None):
    cuda, cudnn = libraries()
    toolchain = ROOT / "output/native-toolchain"
    if crt_include is None:
        crt_include = next(
            (
                path
                for path in (cuda / "include", toolchain / "nvidia/cu13/include")
                if (path / "crt/mma.h").is_file()
            ),
            None,
        )
    if crt_include is None or not (crt_include / "crt/mma.h").is_file():
        raise RuntimeError(
            "Install the matching nvidia-cuda-crt wheel under output/native-toolchain or pass --crt-include"
        )
    if cccl_include is None:
        cccl_include = next(
            (
                path
                for path in (cuda / "include", toolchain / "cccl/nvidia/cu13/include")
                if (path / "nv/target").is_file()
            ),
            None,
        )
    if cccl_include is None or not (cccl_include / "nv/target").is_file():
        raise RuntimeError(
            "Install nvidia-cuda-cccl under output/native-toolchain/cccl or pass --cccl-include"
        )
    env = compiler_environment()
    output.mkdir(parents=True, exist_ok=True)
    compile_kernels(cuda, crt_include, output / "kernels.cubin", arch)
    cpp = SOURCE / "runner.cpp"
    obj = output / "runner.obj"
    subprocess.run(
        [
            "cl.exe",
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/MD",
            "/c",
            "/wd4996",
            f"/I{cuda / 'include'}",
            f"/I{cudnn / 'include'}",
            f"/I{crt_include}",
            f"/I{cccl_include}",
            f"/Fo{obj}",
            str(cpp),
        ],
        env=env,
        cwd=output,
        check=True,
    )
    # The wheels ship these DLLs and headers but omit their MSVC import libraries.
    host_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (cpp, *sorted(SOURCE.glob("*.h")))
    )
    imports = []
    for dll, symbol in (("cudnn64_9", r"cudnn\w+"), ("cublasLt64_13", r"cublasLt\w+")):
        names = sorted(set(re.findall(rf"\b({symbol})\(", host_sources)))
        if dll.startswith("cublas"):
            names.append("cublasLtMatmulPreferenceDestroy")
        definition = output / f"{dll}.def"
        definition.write_text(
            f"LIBRARY {dll}.dll\nEXPORTS\n" + "\n".join(names) + "\n", encoding="ascii"
        )
        library = output / f"{dll}.lib"
        subprocess.run(
            [
                "lib.exe",
                "/nologo",
                "/machine:x64",
                f"/def:{definition}",
                f"/out:{library}",
            ],
            env=env,
            cwd=output,
            check=True,
        )
        imports.append(str(library))
    link = [
        "link.exe",
        "/nologo",
        str(obj),
        str(cuda / "lib/x64/cuda.lib"),
        "bcrypt.lib",
        "version.lib",
        *imports,
    ]
    subprocess.run(
        [*link, "/DLL", f"/OUT:{output / 'gaterv3.dll'}"],
        env=env,
        cwd=output,
        check=True,
    )
    subprocess.run(
        [
            *link,
            f"/OUT:{output / 'gaterv3.exe'}",
            f"/IMPLIB:{output / 'gaterv3-cli.lib'}",
        ],
        env=env,
        cwd=output,
        check=True,
    )
    print(f"Built {output / 'gaterv3.exe'} and {output / 'gaterv3.dll'} for {arch}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/native")
    parser.add_argument("--arch", default="sm_86", help="NVRTC CUDA target, e.g. sm_86")
    parser.add_argument("--crt-include", type=Path)
    parser.add_argument("--cccl-include", type=Path)
    args = parser.parse_args()
    build(args.output.resolve(), args.arch, args.crt_include, args.cccl_include)


if __name__ == "__main__":
    main()
