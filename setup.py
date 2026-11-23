"""Setuptools compatibility shim that prevents stale modules in wheels."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class CleanBuildPy(_build_py):
    """Remove an old package copy before populating ``build/lib``.

    Setuptools normally updates that directory incrementally. After a module is
    deleted or moved, its old copy can therefore leak into a later wheel. Only
    the generated AutoDelphiRF package directory is removed here.
    """

    def run(self):
        package_build_dir = Path(self.build_lib) / "autodelphirf"
        if package_build_dir.is_dir():
            shutil.rmtree(package_build_dir)
        super().run()


setup(cmdclass={"build_py": CleanBuildPy})
