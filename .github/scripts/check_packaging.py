"""Build a source distribution and test its wheel across NumPy ABIs."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
from email.parser import BytesParser
from functools import partial
from pathlib import Path

from build import ProjectBuilder, env

ROOT = Path(__file__).resolve().parents[2]
BUILD_VERSION = "0.dev0+packaging.1"
VERSION_ENV = "PYSPTK_BUILD_VERSION"
NUMPY_REQUIREMENTS = ("numpy==1.26.4", "numpy>=2,<3")

BUILD_VERSIONS = """
from importlib.metadata import version
from platform import machine, python_version

versions = {name: version(name) for name in ("numpy", "Cython", "setuptools")}
print("Build environment:", python_version(), machine(), versions, flush=True)
assert versions["numpy"].split(".")[0] == "2", versions
assert versions["Cython"].split(".")[0] == "3", versions
"""

TEST_WHEEL = r"""
import sys
import sysconfig
import warnings
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

abi_warning = (
    r".*(binary incompatibility|size changed|(C-API|ABI) version|"
    r"compiled (against|using) NumPy|_ARRAY_API).*"
)
warnings.filterwarnings("error", message=abi_warning)

import pysptk
import pytest
from pysptk import _sptk
from pysptk.version import __version__
from scipy.io import wavfile

site_packages = Path(sysconfig.get_path("platlib")).resolve()
package = site_packages / "pysptk"
assert Path(pysptk.__file__).resolve() == package / "__init__.py", pysptk.__file__
extension = Path(_sptk.__file__).resolve()
assert extension.parent == package and extension.suffix in {".so", ".pyd"}, extension
assert pysptk.__version__ == __version__ == version("pysptk") == sys.argv[1]
assert find_spec("Cython") is None
assert find_spec("setuptools") is None
assert find_spec("pkg_resources") is None
assert wavfile.read(pysptk.util.example_audio_file())[0] == 16000

versions = {name: version(name) for name in ("pysptk", "numpy", "scipy", "decorator", "pytest", "pip")}
expected_numpy = sys.argv[2]
if expected_numpy == "numpy==1.26.4":
    assert versions["numpy"] == "1.26.4", versions
else:
    assert versions["numpy"].split(".")[0] == "2", versions
print("Runtime environment:", versions, flush=True)
print("Installed extension:", extension, flush=True)
raise SystemExit(pytest.main([
    "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
    "-o", f"filterwarnings=error:{abi_warning}", sys.argv[3],
]))
"""


def copy_checkout(source):
    paths = subprocess.check_output(
        ["git", "ls-files", "--recurse-submodules", "-z"], cwd=ROOT
    )
    for name in paths.split(b"\0")[:-1]:
        relative = Path(os.fsdecode(name))
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    # Exercise the manifest with a stale generated file present.
    (source / "pysptk" / "_sptk.c").write_text(
        "#error Stale generated source\n", encoding="utf-8"
    )


def run_backend(command, cwd=None, extra_environ=None, *, version=None):
    env = {**os.environ, **(extra_environ or {})}
    env.pop(VERSION_ENV, None)
    if version is not None:
        env[VERSION_ENV] = version
    subprocess.run(command, cwd=cwd, env=env, check=True)


def inspect_sdist(sdist, source, scratch):
    root_name = sdist.name.removesuffix(".tar.gz")
    unpacked = scratch / root_name
    required = {
        "LICENSE.md",
        "pysptk/_sptk.pyx",
        "pysptk/_sptk.pxd",
        "pysptk/example_audio_data/arctic_a0007.wav",
        "pysptk/example_audio_data/COPYING",
        "lib/SPTK/include/SPTK.h",
        "lib/SPTK/COPYING",
    }
    required.update(
        path.relative_to(source).as_posix()
        for path in (source / "lib" / "SPTK").rglob("*")
        if path.is_file() and (path.suffix in {".c", ".h"} or path.name == "COPYING")
    )
    with tarfile.open(sdist, "r:gz") as archive:
        files = {member.name for member in archive if member.isfile()}
        missing = {f"{root_name}/{name}" for name in required} - files
        assert not missing, f"Missing sdist files: {sorted(missing)}"
        assert f"{root_name}/pysptk/_sptk.c" not in files
        archive.extractall(scratch, filter="data")

    with (unpacked / "PKG-INFO").open("rb") as metadata:
        assert (
            BytesParser().parse(metadata, headersonly=True)["Version"] == BUILD_VERSION
        )
    git_probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=unpacked,
        capture_output=True,
        check=False,
    )
    assert git_probe.returncode != 0, f"sdist extracted into a Git worktree: {unpacked}"
    print(f"sdist contains all {len(required)} required files", flush=True)
    return unpacked


def build_distributions(source, scratch):
    dist = scratch / "dist"
    with env.DefaultIsolatedEnv() as isolated:
        builder = ProjectBuilder.from_isolated_env(
            isolated, source, runner=partial(run_backend, version=BUILD_VERSION)
        )
        isolated.install(builder.build_system_requires)
        isolated.install(builder.get_requires_for_build("sdist"))
        sdist = Path(builder.build("sdist", dist))

        unpacked = inspect_sdist(sdist, source, scratch)
        builder = ProjectBuilder.from_isolated_env(
            isolated, unpacked, runner=run_backend
        )
        isolated.install(builder.get_requires_for_build("wheel"))
        subprocess.run(
            [isolated.python_executable, "-I", "-c", BUILD_VERSIONS],
            cwd=scratch,
            check=True,
        )
        wheel = Path(builder.build("wheel", dist))
    subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", sdist, wheel],
        cwd=scratch,
        check=True,
    )
    return sdist, wheel


def test_wheel(wheel, tests, scratch):
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    for index, numpy_requirement in enumerate(NUMPY_REQUIREMENTS):
        directory = scratch / f"runtime-{index}"
        venv.create(directory, with_pip=True)
        python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        pip = [python, "-I", "-m", "pip"]
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == digest
        print(f"{numpy_requirement} | SHA-256 {digest} | {wheel.name}", flush=True)
        subprocess.run(
            [*pip, "install", f"{wheel}[test]", numpy_requirement],
            cwd=scratch,
            check=True,
        )
        subprocess.run([*pip, "check"], cwd=scratch, check=True)
        subprocess.run(
            [
                python,
                "-I",
                "-B",
                "-c",
                TEST_WHEEL,
                BUILD_VERSION,
                numpy_requirement,
                tests,
            ],
            cwd=scratch,
            check=True,
        )
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir", type=Path, help="Keep the verified sdist and wheel."
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="pysptk-packaging-") as directory:
        scratch = Path(directory).resolve()
        source = scratch / "checkout"
        copy_checkout(source)
        sdist, wheel = build_distributions(source, scratch)
        test_wheel(wheel, source / "tests", scratch)
        if args.dist_dir:
            args.dist_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sdist, args.dist_dir)
            shutil.copy2(wheel, args.dist_dir)


if __name__ == "__main__":
    main()
