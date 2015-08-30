# coding: utf-8

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os
from glob import glob
from os.path import join

src_top = "lib/SPTK"

swipe_src = [
    join(src_top, 'bin/pitch/swipe/swipe.c'),
    join(src_top, 'bin/pitch/swipe/vector.c'),
]
hts_engine_src = glob(join(src_top, 'bin/vc/hts_engine_API/*.c'))

sptklib_src = glob(join(src_top, 'lib/*.c'))
sptk_src = glob(join(src_top, 'bin/*/_*.c'))

sptk_all_src = sptk_src + sptklib_src + swipe_src + hts_engine_src

# TODO:
libraries = ["SPTK"]
use_system_lib = False
if use_system_lib:
    sptk_all_src = []
    libraries = []


ext_modules = [Extension(
    name="pysptk.sptk",
    sources=["pysptk/sptk.pyx"] + sptk_all_src,
    include_dirs=[np.get_include(), join(os.getcwd(), 'lib/SPTK/include')],
    libraries=libraries,
    language="c",
)]

setup(
    name='pysptk',
    version='0.0.1',
    description='A python wrapper for SPTK',
    author='Ryuichi Yamamoto',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    install_requires=[
        'numpy >= 1.8.0',
        'cython',
        'six'
    ],
    tests_require=['nose'],
)
