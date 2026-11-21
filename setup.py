from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        package = Path(__file__).parent / "src" / "umzi_native"
        if any((package / name).is_file() for name in ("gaterv3.dll", "libgaterv3.so")):
            self.root_is_pure = False

    def get_tag(self):
        python, abi, platform = super().get_tag()
        if self.root_is_pure:
            return python, abi, platform
        return "py3", "none", platform


setup(cmdclass={"bdist_wheel": PlatformWheel})
