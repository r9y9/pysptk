"""Report and check the isolated build environment (run with the build Python)."""

from importlib.metadata import version
from platform import machine, python_version

versions = {name: version(name) for name in ("numpy", "Cython", "setuptools")}
print("Build environment:", python_version(), machine(), versions, flush=True)
assert versions["numpy"].split(".")[0] == "2", versions
assert versions["Cython"].split(".")[0] == "3", versions
