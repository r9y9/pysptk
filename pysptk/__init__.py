# coding: utf-8

"""
A python wrapper for Speech Signal Processing Toolkit (SPTK)

Note that the wrapper is based on a modified version of SPTK:
https://github.com/r9y9/SPTK
"""
from __future__ import print_function


def assert_gamma(gamma):
    if not (-1 <= gamma <= 0.0):
        raise ValueError("unsupported gamma: must be -1 <= gamma <= 0")

def assert_pade(pade):
    if pade != 4 and pade != 5:
        raise ValueError("4 or 5 pade approximation is supported")

def assert_fftlen(fftlen):
    if fftlen % 2 > 0:
        raise ValueError("fftlen must be power of 2")


from sptk import *