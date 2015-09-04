# coding: utf-8

"""
A python wrapper for `Speech Signal Processing Toolkit (SPTK)
<http://sp-tk.sourceforge.net>`_.

Full documentation
------------------

A full documentation of SPTK is available at http://sp-tk.sourceforge.net.

The wrapper is based on a modified version of SPTK (`r9y9/SPTK
<https://github.com/r9y9/SPTK>`_)
"""

from __future__ import print_function
from __future__ import absolute_import

import pkg_resources

__version__ = pkg_resources.get_distribution('pysptk').version


def assert_gamma(gamma):
    if not (-1 <= gamma <= 0.0):
        raise ValueError("unsupported gamma: must be -1 <= gamma <= 0")


def assert_pade(pade):
    if pade != 4 and pade != 5:
        raise ValueError("4 or 5 pade approximation is supported")


def ispow2(num):
    return ((num & (num - 1)) == 0) and num != 0


def assert_fftlen(fftlen):
    if not ispow2(fftlen):
        raise ValueError("fftlen must be power of 2")


from .sptk import *  # pylint: disable=wildcard-import
