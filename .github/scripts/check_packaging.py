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

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
BUILD_VERSION = "0.dev0+packaging.1"
VERSION_ENV = "PYSPTK_BUILD_VERSION"
NUMPY_REQUIREMENTS = ("numpy==1.26.4", "numpy>=2,<3")


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
            [isolated.python_executable, "-I", SCRIPTS / "verify_build_env.py"],
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
                SCRIPTS / "verify_wheel.py",
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
