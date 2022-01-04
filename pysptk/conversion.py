# coding: utf-8

"""
Other conversions
-----------------

Not exist in SPTK itself, but can be used with the core API.
Functions in the ``pysptk.conversion`` module can also be directly accesible by ``pysptk.*``.

.. autosummary::
    :toctree: generated/

    mgc2b
    sp2mc
    mc2sp
    mc2e
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from pysptk.sptk import c2ir, freqt, gnorm, mc2b
from pysptk.util import apply_along_last_axis, automatic_type_conversion


@apply_along_last_axis
@automatic_type_conversion
def mgc2b(mgc, alpha=0.35, gamma=0.0):
    """Mel-generalized cepstrum to MGLSA filter coefficients

    Parameters
    ----------
    mgc : array, shape
        Mel-generalized cepstrum

    alpha : float
        All-pass constant. Default is 0.35.

    gamma : float
        Parameter of generalized log function. Default is 0.0.

    Returns
    -------
    b : array, shape(same as ``mgc``)
        MGLSA filter coefficients

    See Also
    --------
    pysptk.sptk.mlsadf
    pysptk.sptk.mglsadf
    pysptk.sptk.mc2b
    pysptk.sptk.b2mc
    pysptk.sptk.mcep
    pysptk.sptk.mgcep

    """

    b = mc2b(mgc, alpha)
    if gamma == 0:
        return b

    b = gnorm(b, gamma)

    b[0] = np.log(b[0])
    b[1:] *= gamma

    return b


@apply_along_last_axis
@automatic_type_conversion
def sp2mc(powerspec, order, alpha):
    """Convert spectrum envelope to mel-cepstrum

    This is a simplified implementation of ``mcep`` for input type
    is 4.

    Parameters
    ----------
    powerspec : array
        Power spectrum

    order : int
        Order of mel-cepstrum

    alpha : float
        All-pass constant.

    Returns
    -------
    mc : array, shape(``order+1``)
        mel-cepstrum

    See Also
    --------
    pysptk.sptk.mcep
    pysptk.conversion.mc2sp

    """

    # |X(ω)|² -> log(|X(ω)²|)
    logperiodogram = np.log(powerspec)

    # transform log-periodogram to real cepstrum
    # log(|X(ω)|²) -> c(m)
    c = np.fft.irfft(logperiodogram)
    c[0] /= 2.0

    # c(m) -> cₐ(m)
    return freqt(c, order, alpha)


@apply_along_last_axis
@automatic_type_conversion
def mc2sp(mc, alpha, fftlen):
    """Convert mel-cepstrum back to power spectrum

    Parameters
    ----------
    mc : array
        Mel-spectrum

    alpha : float
        All-pass constant.

    fftlen : int
        FFT length

    Returns
    -------
    powerspec : array, shape(``fftlen//2 +1``)
        Power spectrum

    See Also
    --------
    pysptk.sptk.mcep
    pysptk.conversion.sp2mc

    """
    # back to cepstrum from mel-cesptrum
    # cₐ(m) -> c(m)
    c = freqt(mc, int(fftlen // 2), -alpha)
    c[0] *= 2.0

    symc = np.zeros(fftlen)
    symc[0] = c[0]
    for i in range(1, len(c)):
        symc[i] = c[i]
        symc[-i] = c[i]

    # back to power spectrum
    # c(m) -> log(|X(ω)|²) -> |X(ω)|²
    return np.exp(np.fft.rfft(symc).real)


@apply_along_last_axis
@automatic_type_conversion
def mc2e(mc, alpha=0.35, irlen=256):
    """Compute energy from mel-cepstrum

    Inspired from hts_engine

    Parameters
    ----------
    mc : array
        Mel-spectrum

    alpha : float
        All-pass constant.

    irlen : int
        IIR filter length

    Returns
    -------
    energy : floating point, scalar
        frame energy
    """
    # back to linear frequency domain
    c = freqt(mc, irlen - 1, -alpha)

    # compute impule response from cepsturm
    ir = c2ir(c, irlen)

    return np.sum(np.abs(ir * ir))
