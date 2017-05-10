# coding: utf-8

"""
Utilities
=========

Audio files
-----------

.. autosummary::
    :toctree: generated/

    example_audio_file


Mel-cepstrum analysis
---------------------

.. autosummary::
    :toctree: generated/

    mcepalpha
"""

from __future__ import division, print_function, absolute_import

import pkg_resources
import numpy as np

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


def mcepalpha(fs, start=0.0, stop=1.0, step=0.001, num_points=1000):
    """Compute appropriate frequency warping parameter given a sampling frequency

    It would be useful to determine alpha parameter in mel-cepstrum analysis.

    The code is traslated from https://bitbucket.org/happyalu/mcep_alpha_calc.

    Parameters
    ----------
    fs : int
        Sampling frequency

    start : float
        start value that will be passed to numpy.arange. Default is 0.0.

    stop : float
        stop value that will be passed to numpy.arange. Default is 1.0.

    step : float
        step value that will be passed to numpy.arange. Default is 0.001.

    num_points : int
        Number of points used in approximating mel-scale vectors in fixed-
        length.

    Returns
    -------
    alpha : float
        frequency warping paramter (offen denoted by alpha)

    See Also
    --------
    pysptk.sptk.mcep
    pysptk.sptk.mgcep

    """
    alpha_candidates = np.arange(start, stop, step)
    mel = _melscale_vector(fs, num_points)
    distances = [rms_distance(mel, _warping_vector(alpha, num_points)) for
                 alpha in alpha_candidates]
    return alpha_candidates[np.argmin(distances)]


def _melscale_vector(fs, length):
    step = (fs / 2.0) / length
    melscalev = 1000.0 / np.log(2) * np.log(1 +
                                            step * np.arange(0, length) / 1000.0)
    return melscalev / melscalev[-1]


def _warping_vector(alpha, length):
    step = np.pi / length
    omega = step * np.arange(0, length)
    num = (1 - alpha * alpha) * np.sin(omega)
    den = (1 + alpha * alpha) * np.cos(omega) - 2 * alpha
    warpfreq = np.arctan(num / den)
    warpfreq[warpfreq < 0] += np.pi
    return warpfreq / warpfreq[-1]


def rms_distance(v1, v2):
    d = v1 - v2
    return np.sum(np.abs(d * d)) / len(v1)
