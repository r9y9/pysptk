# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages, Extension
import setuptools.command.develop
import setuptools.command.build_py
from distutils.version import LooseVersion

import numpy as np
import os
from glob import glob
from os.path import join
import subprocess

version = '0.1.13'

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv('PYSPTK_BUILD_VERSION'):
    version = os.getenv('PYSPTK_BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
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
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'pysptk', 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):

    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


cmdclass = {"build_py": build_py, "develop": develop}

min_cython_ver = '0.21.0'
try:
    import Cython
    ver = Cython.__version__
    _CYTHON_INSTALLED = ver >= LooseVersion(min_cython_ver)
except ImportError:
    _CYTHON_INSTALLED = False

try:
    if not _CYTHON_INSTALLED:
        raise ImportError('No supported version of Cython installed.')
    from Cython.Distutils import build_ext
    cython = True
except ImportError:
    cython = False

if cython:
    ext = '.pyx'
    cmdclass['build_ext'] = build_ext
else:
    ext = '.c'
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
ignore_bin_list = [join(src_bin_top, "wavjoin"), join(src_bin_top, "wavsplit"),
                   join(src_bin_top, "vc")]
for ignore in ignore_bin_list:
    sptk_all_src = list(
        filter(lambda s: not s.startswith(ignore), sptk_all_src))

# define core cython module
ext_modules = [Extension(
    name="pysptk._sptk",
    sources=[join("pysptk", "_sptk" + ext)] + sptk_all_src,
    include_dirs=[np.get_include(), join(
        os.getcwd(), "lib", "SPTK", "include")],
    language="c",
    extra_compile_args=['-std=c99']
)]

setup(
    name='pysptk',
    version=version,
    description='A python wrapper for Speech Signal Processing Toolkit (SPTK)',
    author='Ryuichi Yamamoto',
    author_email='zryuichi@gmail.com',
    url='https://github.com/r9y9/pysptk',
    license='MIT',
    packages=find_packages(),
    package_data={'': ['example_audio_data/*']},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'numpy >= 1.8.0',
        'six',
        'decorator'
    ],
    tests_require=['nose', 'coverage'],
    extras_require={
        'docs': ['numpydoc', 'sphinx_rtd_theme', 'seaborn'],
        'test': ['nose'],
        'develop': ['cython >= ' + min_cython_ver],
    },
    classifiers=[
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    keywords=["SPTK"]
)
