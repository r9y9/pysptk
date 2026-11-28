"""Verify an installed wheel and run the test suite against it.

Usage: verify_wheel.py EXPECTED_VERSION NUMPY_REQUIREMENT TESTS_DIR
"""

import sys
import sysconfig
import warnings
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

ABI_WARNING = (
    r".*(binary incompatibility|size changed|(C-API|ABI) version|"
    r"compiled (against|using) NumPy|_ARRAY_API).*"
)
# Must be active before importing the extension module.
warnings.filterwarnings("error", message=ABI_WARNING)

import pysptk  # noqa: E402
import pytest  # noqa: E402
from pysptk import _sptk  # noqa: E402
from pysptk.version import __version__  # noqa: E402
from scipy.io import wavfile  # noqa: E402

PACKAGES = ("pysptk", "numpy", "scipy", "decorator", "pytest", "pip")


def check_installation(expected_version):
    site_packages = Path(sysconfig.get_path("platlib")).resolve()
    package = site_packages / "pysptk"
    assert Path(pysptk.__file__).resolve() == package / "__init__.py", pysptk.__file__
    extension = Path(_sptk.__file__).resolve()
    assert extension.parent == package, extension
    assert extension.suffix in {".so", ".pyd"}, extension
    assert pysptk.__version__ == __version__ == version("pysptk") == expected_version
    for build_only in ("Cython", "setuptools", "pkg_resources"):
        assert find_spec(build_only) is None, build_only
    assert wavfile.read(pysptk.util.example_audio_file())[0] == 16000
    print("Installed extension:", extension, flush=True)


def check_numpy(requirement):
    versions = {name: version(name) for name in PACKAGES}
    if requirement == "numpy==1.26.4":
        assert versions["numpy"] == "1.26.4", versions
    else:
        assert versions["numpy"].split(".")[0] == "2", versions
    print("Runtime environment:", versions, flush=True)


def main():
    expected_version, numpy_requirement, tests = sys.argv[1:]
    check_installation(expected_version)
    check_numpy(numpy_requirement)
    return pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
            "-o",
            f"filterwarnings=error:{ABI_WARNING}",
            tests,
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
