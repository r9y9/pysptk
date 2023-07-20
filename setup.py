import os
import subprocess
from distutils.version import LooseVersion
from glob import glob
from os.path import join

import setuptools.command.build_py
import setuptools.command.develop
from setuptools import Extension, find_packages, setup

version = "0.2.1"

# Adapted from https://github.com/py_torch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv("PYSPTK_BUILD_VERSION"):
    version = os.getenv("PYSPTK_BUILD_VERSION")
else:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("ascii")
            .strip()
        )
        version += "+" + sha[:7]
    except subprocess.CalledProcessError:
        pass
    except IOError:  # FileNotFoundError for python 3
        pass


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, cwd
        print("-- Building version " + version)
        version_path = os.path.join(cwd, "pysptk", "version.py")
        with open(version_path, "w") as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):
    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


cmdclass = {"build_py": build_py, "develop": develop}

min_cython_ver = "0.28.0"
try:
    import Cython

    ver = Cython.__version__
    _CYTHON_INSTALLED = ver >= LooseVersion(min_cython_ver)
except ImportError:
    _CYTHON_INSTALLED = False

try:
    if not _CYTHON_INSTALLED:
        raise ImportError("No supported version of Cython installed.")
    from Cython.Distutils import build_ext

    cython = True
except ImportError:
    cython = False
    from setuptools.command.build_ext import build_ext as _build_ext

    class build_ext(_build_ext):
        # https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py # noqa
        def finalize_options(self):
            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy

            self.include_dirs.append(numpy.get_include())


include_dirs = [join(os.getcwd(), "lib", "SPTK", "include")]
cmdclass["build_ext"] = build_ext
if cython:
    ext = ".pyx"
    import numpy as np

    include_dirs.insert(0, np.get_include())
else:
    ext = ".c"
    if not os.path.exists(join("pysptk", "_sptk" + ext)):
        raise RuntimeError("Cython is required to generate C code.")

# SPTK sources
src_top = join("lib", "SPTK")
src_bin_top = join(src_top, "bin")
swipe_src = [
    join(src_bin_top, "pitch", "swipe", "swipe.c"),
    join(src_bin_top, "pitch", "swipe", "vector.c"),
]
rapt_src = [
    join(src_bin_top, "pitch", "snack", "jkGetF0.c"),
    join(src_bin_top, "pitch", "snack", "sigproc.c"),
]

sptklib_src = glob(join(src_top, "lib", "*.c"))
sptk_src = glob(join(src_bin_top, "*", "_*.c"))

# collect all sources
sptk_all_src = sptk_src + sptklib_src + swipe_src + rapt_src

# Filter ignore list
ignore_bin_list = [
    join(src_bin_top, "wavjoin"),
    join(src_bin_top, "wavsplit"),
    join(src_bin_top, "vc"),
]
for ignore in ignore_bin_list:
    sptk_all_src = list(filter(lambda s, ig=ignore: not s.startswith(ig), sptk_all_src))

# define core cython module
ext_modules = [
    Extension(
        name="pysptk._sptk",
        sources=[join("pysptk", "_sptk" + ext)] + sptk_all_src,
        include_dirs=include_dirs,
        language="c",
        extra_compile_args=["-std=c99"],
    )
]

with open("README.md", "r") as fh:
    LONG_DESC = fh.read()

setup(
    name="pysptk",
    version=version,
    description="A python wrapper for Speech Signal Processing Toolkit (SPTK)",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Ryuichi Yamamoto",
    author_email="zryuichi@gmail.com",
    url="https://github.com/r9y9/pysptk",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples"]),
    package_data={"": ["example_audio_data/*"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    setup_requires=["numpy >= 1.20.0"],
    install_requires=[
        "scipy",
        "decorator",
        "cython >= " + min_cython_ver,
    ],
    tests_require=["pytest", "pytest-cov", "coverage"],
    extras_require={
        "docs": ["numpydoc", "sphinx_rtd_theme", "seaborn"],
        "test": ["pytest", "pytest-cov", "coverage", "flake8"],
        "lint": [
            "pysen",
            "types-setuptools",
            "mypy<=0.910",
            "black>=19.19b0,<=20.8",
            "click<8.1.0",
            "flake8>=3.7,<4",
            "flake8-bugbear",
            "isort>=4.3,<5.2.0",
            "types-decorator",
            "importlib-metadata<5.0",
        ],
    },
    classifiers=[
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    keywords=["SPTK"],
)
