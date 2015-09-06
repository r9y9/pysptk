# coding: utf-8

"""
Utilities
=========

Audio files
-----------

.. autosummary::
    :toctree: generated/

    example_audio_file

"""

from __future__ import division, print_function, absolute_import

import pkg_resources

# 16kHz, 16bit example audio from cmu_us_awb_arctic
# see COPYING for the license of the audio file.
EXAMPLE_AUDIO = 'example_audio_data/arctic_a0007.wav'


def assert_gamma(gamma):
    if not (-1 <= gamma <= 0.0):
        raise ValueError("unsupported gamma: must be -1 <= gamma <= 0")


def assert_pade(pade):
    if pade != 4 and pade != 5:
        raise ValueError("4 or 5 pade approximation is supported")


def assert_stage(stage):
    if stage < 1:
        raise ValueError("stage >= 1 (-1 <= gamma < 0)")


def ispow2(num):
    return ((num & (num - 1)) == 0) and num != 0


def assert_fftlen(fftlen):
    if not ispow2(fftlen):
        raise ValueError("fftlen must be power of 2")


def example_audio_file():
    """Get the path to an included audio example file.

    Examples
    --------
    >>> from scipy.io import wavfile
    >>> fs, x = wavfile.read(pysptk.util.example_audio_file())

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, label="cmu_us_awb_arctic arctic_a0007.wav")
    >>> plt.xlim(0, len(x))
    >>> plt.legend()

    """

    return pkg_resources.resource_filename(__name__, EXAMPLE_AUDIO)
