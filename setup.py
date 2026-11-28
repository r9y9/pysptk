import os
import subprocess
from email.parser import BytesParser
from pathlib import Path

from Cython.Distutils import build_ext as cython_build_ext
from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).resolve().parent
DEV_VERSION = "1.0.2.dev0"


def get_version():
    version = os.environ.get("PYSPTK_BUILD_VERSION")
    if version:
        return version

    sdist_metadata = ROOT / "PKG-INFO"
    if sdist_metadata.is_file():
        with sdist_metadata.open("rb") as metadata:
            return BytesParser().parse(metadata, headersonly=True)["Version"]

    if (ROOT / ".git").exists():
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return DEV_VERSION
        return f"{DEV_VERSION}+{sha}"

    return DEV_VERSION


class build_ext(cython_build_ext):
    def build_extensions(self):
        import numpy

        for extension in self.extensions:
            extension.include_dirs.append(numpy.get_include())
            if self.compiler.compiler_type != "msvc":
                extension.extra_compile_args.append("-std=c99")
        super().build_extensions()


sptk = Path("lib", "SPTK")
include_dir = sptk / "include"
if not (ROOT / include_dir / "SPTK.h").is_file():
    raise RuntimeError("SPTK sources are missing. Run git submodule update --init.")

bin_dir = sptk / "bin"
excluded = {"wavjoin", "wavsplit", "vc"}
sources = [
    *sorted(
        path for path in bin_dir.glob("*/_*.c") if path.parent.name not in excluded
    ),
    *sorted((sptk / "lib").glob("*.c")),
    bin_dir / "pitch" / "swipe" / "swipe.c",
    bin_dir / "pitch" / "swipe" / "vector.c",
    bin_dir / "pitch" / "snack" / "jkGetF0.c",
    bin_dir / "pitch" / "snack" / "sigproc.c",
]

setup(
    name="pysptk",
    version=get_version(),
    description="A python wrapper for Speech Signal Processing Toolkit (SPTK)",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Ryuichi Yamamoto",
    author_email="zryuichi@gmail.com",
    url="https://github.com/r9y9/pysptk",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples"]),
    package_data={"pysptk": ["example_audio_data/*"]},
    ext_modules=[
        Extension(
            name="pysptk._sptk",
            sources=["pysptk/_sptk.pyx", *(str(path) for path in sources)],
            include_dirs=[str(ROOT / include_dir)],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                ("NPY_TARGET_VERSION", "NPY_1_20_API_VERSION"),
            ],
            language="c",
        )
    ],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=["numpy>=1.20", "scipy", "decorator"],
    extras_require={
        "docs": ["sphinx", "numpydoc", "sphinx_rtd_theme", "seaborn"],
        "test": ["pytest", "pytest-cov"],
        "lint": ["ruff"],
    },
    classifiers=[
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    keywords=["SPTK"],
)
