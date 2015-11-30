# coding: utf-8

"""
Library routines
----------------
.. autosummary::
    :toctree: generated/

    agexp
    gexp
    glog
    mseq

Adaptive cepstrum analysis
--------------------------
.. autosummary::
    :toctree: generated/

    acep
    agcep
    amcep

Mel-generalized cepstrum analysis
---------------------------------
.. autosummary::
    :toctree: generated/

    mcep
    gcep
    mgcep
    uels
    fftcep
    lpc

MFCC
----
.. autosummary::
    :toctree: generated/

    mfcc

LPC, LSP and PARCOR conversions
-------------------------------
.. autosummary::
    :toctree: generated/

    lpc2c
    lpc2lsp
    lpc2par
    par2lpc
    lsp2sp

Mel-generalized cepstrum conversions
------------------------------------
.. autosummary::
    :toctree: generated/

    mc2b
    b2mc
    c2acr
    c2ir
    ic2ir
    c2ndps
    ndps2c
    gc2gc
    gnorm
    ignorm
    freqt
    mgc2mgc
    mgc2sp
    mgclsp2sp

F0 analysis
-----------
.. autosummary::
    :toctree: generated/

    swipe
    rapt

Excitation generation
---------------------
.. autosummary::
    :toctree: generated/

    excite

Window functions
----------------
.. autosummary::
    :toctree: generated/

    blackman
    hamming
    hanning
    bartlett
    trapezoid
    rectangular

Waveform generation filters
---------------------------
.. autosummary::
    :toctree: generated/

    poledf
    lmadf
    lspdf
    ltcdf
    glsadf
    mlsadf
    mglsadf

Utilities for waveform generation filters
-----------------------------------------
.. autosummary::
    :toctree: generated/

    poledf_delay
    lmadf_delay
    lspdf_delay
    ltcdf_delay
    glsadf_delay
    mlsadf_delay
    mglsadf_delay

"""

import numpy as np

from . import _sptk


### Library routines ###

def agexp(r, x, y):
    """Magnitude squared generalized exponential function

    Parameters
    ----------
    r : float
        Gamma
    x : float
        Real part
    y : float
        Imaginary part

    Returns
    -------
    Value

    """
    return _sptk.agexp(r, x, y)


def gexp(r, x):
    """Generalized exponential function

    Parameters
    ----------
    r : float
        Gamma

    x : float
        Arg

    Returns
    -------
    Value

    """
    return _sptk.gexp(r, x)


def glog(r, x):
    """Generalized logarithmic function

    Parameters
    ----------
    r : float
        Gamma
    x : float
        Arg

    Returns
    -------
    Value

    """
    return _sptk.glog(r, x)


def mseq():
    """M-sequence

    Returns
    -------
    A sample of m-sequence

    """
    return _sptk.mseq()


### Adaptive mel-generalized cepstrum analysis ###

def acep(x, c, lambda_coef=0.98, step=0.1, tau=0.9, pd=4, eps=1.0e-6):
    """Adaptive cepstral analysis

    Parameters
    ----------
    x : double
        A input sample

    c : array, shape(``order + 1``)
        Cepstrum. The result is stored in place.

    lambda_coef : float, optional
        Leakage factor. Default is 0.98.

    step : float, optional
        Step size. Default is 0.1.

    tau : float, optional
        Momentum constant. Default is 0.9.

    pd : int, optional
        Order of pade approximation. Default is 4.

    eps : float, optional
        Minimum value for epsilon. Default is 1.0e-6.

    Returns
    -------
    prederr : float
        Prediction error

    Raises
    ------
    ValueError
        if invalid order of pade approximation is specified

    See Also
    --------
    pysptk.sptk.uels
    pysptk.sptk.gcep
    pysptk.sptk.mcep
    pysptk.sptk.mgcep
    pysptk.sptk.amcep
    pysptk.sptk.agcep
    pysptk.sptk.lmadf

    """
    return _sptk.acep(x, c, lambda_coef, step, tau, pd, eps)


def agcep(x, c, stage=1, lambda_coef=0.98, step=0.1, tau=0.9, eps=1.0e-6):
    """Adaptive generalized cepstral analysis

    Parameters
    ----------
    x : float
        A input sample

    c : array, shape(``order + 1``), optional
        Cepstrum. The result is stored in-place.

    stage : int, optional
        -1 / gamma. Default is 1.

    lambda_coef : float, optional
        Leakage factor. Default is 0.98.

    step : float, optional
        Step size. Default is 0.1.

    tau : float, optional
        Momentum constant. Default is 0.9.

    eps : float, optional
        Minimum value for epsilon. Default is 1.0e-6.

    Returns
    -------
    prederr : float
        Prediction error

    Raises
    ------
    ValueError
        if invalid number of stage is specified

    See Also
    --------
    pysptk.sptk.acep
    pysptk.sptk.amcep
    pysptk.sptk.glsadf

    """
    return _sptk.agcep(x, c, stage, lambda_coef, step, tau, eps)


def amcep(x, b, alpha=0.35, lambda_coef=0.98, step=0.1, tau=0.9, pd=4, eps=1.0e-6):
    """Adaptive mel-cepstral analysis

    Parameters
    ----------
    x : float
        A input sample

    b : array, shape(``order + 1``), optional
        MLSA filter coefficients. The result is stored in-place.

    alpha : float, optional
        All-pass constant. Default is 0.35.

    lambda_coef : float, optional
        Leakage factor. Default is 0.98.

    step : float, optional
        Step size. Default is 0.1.

    tau : float, optional
        Momentum constant. Default is 0.9.

    pd : int, optional
        Order of pade approximation. Default is 4.

    eps : float, optional
        Minimum value for epsilon. Default is 1.0e-6.

    Returns
    -------
    prederr : float
        Prediction error

    Raises
    ------
    ValueError
        if invalid order of pade approximation is specified

    See Also
    --------
    pysptk.sptk.acep
    pysptk.sptk.agcep
    pysptk.sptk.mc2b
    pysptk.sptk.b2mc
    pysptk.sptk.mlsadf

    """
    return _sptk.amcep(x, b, alpha, lambda_coef, step, tau, pd, eps)


### Mel-generalized cepstrum analysis ###

def mcep(windowed,
         order=25, alpha=0.35,
         miniter=2,
         maxiter=30,
         threshold=0.001,
         etype=0,
         eps=0.0,
         min_det=1.0e-6,
         itype=0):
    """Mel-cepstrum analysis

    Parameters
    ----------
    windowed : array, shape (``frame_len``)
        A windowed frame

    order : int, optional
        Order of mel-cepstrum. Default is 25.

    alpha : float, optional
        All pass constant. Default is 0.35.

    miniter : int, optional
        Minimum number of iteration. Default is 2.

    maxiter : int, optional
        Maximum number of iteration. Default is 30.

    threshold : float, optional
        Threshold in theq. Default is 0.001.

    etype : int, optional
        Type of parameter ``eps``
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

        Default is 0.

    eps : float, optional
        Initial value for log-periodogram or floor of periodogram in db.
        Default is 0.0.

    min_det : float, optional
        Mimimum value of the determinant of normal matrix.
        Default is 1.0e-6

    itype : float, optional
        Input data type
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

        Default is 0.

    Returns
    -------
    mc : array, shape (``order + 1``)
        Mel-cepstrum

    Raises
    ------
    ValueError
        - if invalid ``itype`` is specified
        - if invalid ``etype`` is specified
        - if nonzero ``eps`` is specified when etype = 0
        - if negative ``eps`` is specified
        - if negative ``min_det`` is specified

    RuntimeError
        - if zero(s) are found in periodogram
        - if error happened in theq

    See Also
    --------
    pysptk.sptk.uels
    pysptk.sptk.gcep
    pysptk.sptk.mgcep
    pysptk.sptk.mlsadf

    """
    return _sptk.mcep(windowed, order, alpha, miniter, maxiter, threshold,
                      etype, eps, min_det, itype)


def gcep(windowed, order=25, gamma=0.0,
         miniter=2,
         maxiter=30,
         threshold=0.001,
         etype=0,
         eps=0.0,
         min_det=1.0e-6,
         itype=0,
         norm=False):
    """Generalized-cepstrum analysis

    Parameters
    ----------
    windowed : array, shape (``frame_len``)
        A windowed frame

    order : int, optional
        Order of generalized-cepstrum. Default is 25.

    gamma : float, optional
        Parameter of generalized log function. Default is 0.0.

    miniter : int, optional
        Minimum number of iteration. Default is 2.

    maxiter : int, optional
        Maximum number of iteration. Default is 30.

    threshold : float, optional
        Threshold in theq. Default is 0.001

    etype : int, optional
        Type of parameter ``eps``
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

        Default is 0.

    eps : float, optional
        Initial value for log-periodogram or floor of periodogram in db.
        Default is 0.0.

    min_det : float, optional
        Mimimum value of the determinant of normal matrix. Default is 1.0e-6.

    itype : float, optional
        Input data type
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

        Default is 0.

    Returns
    -------
    gc : array, shape (``order + 1``)
        Generalized cepstrum

    Raises
    ------
    ValueError
        - if invalid ``itype`` is specified
        - if invalid ``etype`` is specified
        - if nonzero ``eps`` is specified when etype = 0
        - if negative ``eps`` is specified
        - if negative ``min_det`` is specified

    RuntimeError
        - if error happened in theq

    See Also
    --------
    pysptk.sptk.uels
    pysptk.sptk.mcep
    pysptk.sptk.mgcep
    pysptk.sptk.glsadf

    """

    return _sptk.gcep(windowed, order, gamma, miniter, maxiter, threshold,
                      etype, eps, min_det, itype)


def mgcep(windowed, order=25, alpha=0.35, gamma=0.0,
          num_recursions=None,
          miniter=2,
          maxiter=30,
          threshold=0.001,
          etype=0,
          eps=0.0,
          min_det=1.0e-6,
          itype=0,
          otype=0):
    """Mel-generalized cepstrum analysis

    Parameters
    ----------
    windowed : array, shape (``frame_len``)
        A windowed frame

    order : int, optional
        Order of mel-generalized cepstrum. Default is 25.

    alpha : float, optional
        All pass constant. Default is 0.35.

    gamma : float, optional
        Parameter of generalized log function. Default is 0.0.

    num_recursions : int, optional
        Number of recursions. Default is ``len(windowed) - 1``.

    miniter : int, optional
        Minimum number of iteration. Default is 2.

    maxiter : int, optional
        Maximum number of iteration. Default is 30.

    threshold : float, optional
        Threshold. Default is 0.001.

    etype : int, optional
        Type of paramter ``e``
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

        Default is 0.

    eps : float, optional
        Initial value for log-periodogram or floor of periodogram in db.
        Default is 0.0.

    min_det : float, optional
        Mimimum value of the determinant of normal matrix.
        Default is 1.0e-6.

    itype : float, optional
        Input data type
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

        Default is 0.

    otype : int, optional
        Output data type
            (0) mel generalized cepstrum: (c~0...c~m)
            (1) MGLSA filter coefficients: b0...bm
            (2) K~,c~'1...c~'m
            (3) K,b'1...b'm
            (4) K~,g*c~'1...g*c~'m
            (5) K,g*b'1...g*b'm

        Default is 0.

    Returns
    -------
    mgc : array, shape (``order + 1``)
        mel-generalized cepstrum

    Raises
    ------
    ValueError
        - if invalid ``itype`` is specified
        - if invalid ``etype`` is specified
        - if nonzero ``eps`` is specified when etype = 0
        - if negative ``eps`` is specified
        - if negative ``min_det`` is specified
        - if invalid ``otype`` is specified

    RuntimeError
        - if error happened in theq

    See Also
    --------
    pysptk.sptk.uels
    pysptk.sptk.gcep
    pysptk.sptk.mcep
    pysptk.sptk.freqt
    pysptk.sptk.gc2gc
    pysptk.sptk.mgc2mgc
    pysptk.sptk.gnorm
    pysptk.sptk.mglsadf

    """

    return _sptk.mgcep(windowed, order, alpha, gamma, num_recursions, miniter,
                       maxiter, threshold, etype, eps, min_det, itype, otype)


def uels(windowed, order=25,
         miniter=2,
         maxiter=30,
         threshold=0.001,
         etype=0,
         eps=0.0,
         itype=0):
    """Unbiased estimation of log spectrum

    Parameters
    ----------
    windowed : array, shape (``frame_len``)
        A windowed frame

    order : int, optional
        Order of cepstrum. Default is 25.

    miniter : int, optional
        Minimum number of iteration. Default is 2.

    maxiter : int, optional
        Maximum number of iteration. Default is 30.

    threshold : float, optional
        Threshold in theq. Default is 0.001

    etype : int, optional
        Type of parameter ``eps``
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

        Default is 0.

    eps : float, optional
        Initial value for log-periodogram or floor of periodogram in db.
        Default is 0.0.

    itype : float, optional
        Input data type
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

        Default is 0.

    Returns
    -------
    c : array, shape (``order + 1``)
        cepstrum estimated by uels

    Raises
    ------
    ValueError
        - if invalid ``itype`` is specified
        - if invalid ``etype`` is specified
        - if nonzero ``eps`` is specified when etype = 0
        - if negative ``eps`` is specified

    RuntimeError
        - if zero(s) are found in periodogram

    See Also
    --------
    pysptk.sptk.gcep
    pysptk.sptk.mcep
    pysptk.sptk.mgcep
    pysptk.sptk.lmadf

    """

    return _sptk.uels(windowed, order, miniter, maxiter, threshold, etype, eps,
                      itype)


def fftcep(logsp,
           order=25,
           num_iter=0,
           acceleration_factor=0.0):
    """FFT-based cepstrum analysis

    Parameters
    ----------
    logsp : array, shape (``frame_len``)
        Log power spectrum

    order : int, optional
        Order of cepstrum. Default is 25.

    num_iter : int, optional
        Number of iteration. Default is 0.

    acceleration_factor : float, optional
        Acceleration factor. Default is 0.0.

    Returns
    -------
    c : array, shape (``order + 1``)
        Cepstrum

    See Also
    --------
    pysptk.sptk.uels

    """

    return _sptk.fftcep(logsp, order, num_iter, acceleration_factor)


def lpc(windowed, order=25, min_det=1.0e-6):
    """Linear prediction analysis

    Parameters
    ----------
    windowed : array, shape (``frame_len``)
        A windowed frame

    order : int, optional
        Order of LPC. Default is 25.

    min_det : float, optional
        Mimimum value of the determinant of normal matrix.
        Default is 1.0e-6.

    Returns
    -------
    a : array, shape (``order + 1``)
        LPC

    Raises
    ------
    ValueError
        - if negative ``min_det`` is specified

    RuntimeError
        - if error happened in levdur


    See Also
    --------
    pysptk.sptk.lpc2par
    pysptk.sptk.par2lpc
    pysptk.sptk.lpc2c
    pysptk.sptk.lpc2lsp
    pysptk.sptk.ltcdf
    pysptk.sptk.lspdf

    """
    return _sptk.lpc(windowed, order, min_det)


### MFCC ###

def mfcc(x, order=14, fs=16000, alpha=0.97, eps=1.0, window_len=None,
         frame_len=None, num_filterbanks=20, cepslift=22, use_dft=False,
         use_hamming=False, czero=False, power=False):
    """MFCC

    Parameters
    ----------
    x : array
        A input signal

    order : int, optional
        Order of MFCC. Default is 14.

    fs : int, optional
        Sampling frequency. Default is 160000.

    alpha : float, optional
        Pre-emphasis coefficient. Default is 0.97.

    eps : float, optional
        Flooring value for calculating ``log(x)`` in filterbank analysis.
        Default is 1.0.

    window_len : int, optional
        Window lenght. Default is ``len(x)``.

    frame_len : int, optional
        Frame length. Default is ``len(x)``.

    num_filterbanks : int, optional
        Number of mel-filter banks. Default is 20.

    cepslift : int, optional
        Liftering coefficient. Default is 22.

    use_dft : bool, optional
        Use DFT (not FFT) or not. Default is False.

    use_hamming : bool, optional
        Use hamming window or not. Default is False.

    czero : bool, optional
        If True, ``mfcc`` returns 0-th coefficient as well. Default is False.

    power : bool, optional
        If True, ``mfcc`` returns power coefficient as well. Default is False.

    Returns
    -------
    cc : array
        MFCC vector, which is ordered as:

        mfcc[0], mfcc[1], mfcc[2], ... mfcc[order-1], c0, Power.

        Note that c0 and Power are optional.

        Shape of ``cc`` is:

            - ``order`` by default.
            - ``orde + 1`` if ``czero`` or ``power`` is set to True.
            - ``order + 2`` if both ``czero`` and ``power`` is set to True.

    Raises
    ------
    ValueError
        if ``num_filterbanks`` is less than or equal to ``order``

    See Also
    --------
    pysptk.sptk.gcep
    pysptk.sptk.mcep
    pysptk.sptk.mgcep

    """

    return _sptk.mfcc(x, order, fs, alpha, eps, window_len,
                      frame_len, num_filterbanks, cepslift, use_dft,
                      use_hamming, czero, power)


### LPC, LSP and PARCOR conversions ###

def lpc2c(lpc, order=None):
    """LPC to cepstrum

    Parameters
    ----------
    lpc : array
        LPC

    order : int, optional
        Order of cepstrum. Default is ``len(lpc) - 1``.

    Returns
    -------
    ceps : array, shape (``order + 1``)
        cepstrum

    See Also
    --------
    pysptk.sptk.lpc
    pysptk.sptk.lspdf

    """

    return _sptk.lpc2c(lpc, order)


def lpc2lsp(lpc, numsp=512, maxiter=4, eps=1.0e-6, loggain=False, otype=0,
            fs=None):
    """LPC to LSP

    Parameters
    ----------
    lpc : array
        LPC

    numsp : int, optional
        Number of unit circle. Default is 512.

    maxiter : int, optional
        Maximum number of iteration. Default is 4.

    eps : float, optional
        End condition for iteration. Default is 1.0e-6.

    loggain : bool, optional
        whether the converted lsp should have loggain or not.
        Default is False.

    fs : int, optional
        Sampling frequency. Default is None and unused.

    otype : int, optional
        Output format LSP
            (0)  normalized frequency (0 ~ pi)
            (1)  normalized frequency (0 ~ 0.5)
            (2)  frequency (kHz)
            (3)  frequency (Hz)

        Default is 0.

    Returns
    -------
    lsp : array, shape (``order + 1``)
        LSP

    raises
    ------
    ValueError
        if ``fs`` is not specified when otype = 2 or 3.

    See Also
    --------
    pysptk.sptk.lpc
    pysptk.sptk.lspdf

    """

    return _sptk.lpc2lsp(lpc, numsp, maxiter, eps, loggain, otype, fs)


def lpc2par(lpc):
    """LPC to PARCOR

    Parameters
    ----------
    lpc : array
        LPC

    Returns
    -------
    par : array, shape (same as ``lpc``)
        PARCOR

    See Also
    --------
    pysptk.sptk.lpc
    pysptk.sptk.par2lpc
    pysptk.sptk.ltcdf

    """

    return _sptk.lpc2par(lpc)


def par2lpc(par):
    """PARCOR to LPC

    Parameters
    ----------
    par : array
        PARCOR

    Returns
    -------
    lpc : array, shape (same as ``par``)
        LPC

    See Also
    --------
    pysptk.sptk.lpc
    pysptk.sptk.lpc2par

    """

    return _sptk.par2lpc(par)


def lsp2sp(lsp, fftlen=256):
    """LSP to spectrum

    Parameters
    ----------
    lsp : array
        LSP

    fftlen : int, optional
        FFT length

    TODO: consider ``otype`` optional argument

    Returns
    -------
    sp : array, shape
        Spectrum. ln|H(z)|.

    Notes
    -----
    It is asuumed that ``lsp`` has loggain at ``lsp[0]``.

    See Also
    --------
    pysptk.sptk.lpc2par

    """

    return _sptk.lsp2sp(lsp, fftlen)


### Mel-generalized cepstrum conversions ###

def mc2b(mc, alpha=0.35):
    """Mel-cepsrum to MLSA filter coefficients

    Parameters
    ----------
    mc : array, shape
        Mel-cepstrum.

    alpha : float, optional
        All-pass constant. Default is 0.35.

    Returns
    -------
    b : array, shape(same as ``mc``)
        MLSA filter coefficients

    See Also
    --------
    pysptk.sptk.mlsadf
    pysptk.sptk.mglsadf
    pysptk.sptk.b2mc
    pysptk.sptk.mcep
    pysptk.sptk.mgcep
    pysptk.sptk.amcep

    """

    return _sptk.mc2b(mc, alpha)


def b2mc(b, alpha=0.35):
    """MLSA filter coefficients to mel-cesptrum

    Parameters
    ----------
    b : array, shape
        MLSA filter coefficients

    alpha : float, optional
        All-pass constant. Default is 0.35.

    Returns
    -------
    mc : array, shape (same as ``b``)
        Mel-cepstrum.

    See Also
    --------
    pysptk.sptk.mc2b
    pysptk.sptk.mcep
    pysptk.sptk.mlsadf

    """

    return _sptk.b2mc(b, alpha)


def b2c(b, dst_order=None, alpha=0.35):
    return _sptk.b2c(b, dst_order, alpha)


def c2acr(c, order=None, fftlen=256):
    """Cepstrum to autocorrelation

    Parameters
    ----------
    c : array
        Cepstrum

    order : int, optional
        Order of cepstrum. Default is ``len(c) - 1``.

    fftlen : int, optional
        FFT length. Default is 256.

    Returns
    -------
    r : array, shape (``order + 1``)
        Autocorrelation

    Raises
    ------
    ValueError
        if non power of 2 ``fftlen`` is specified

    See Also
    --------
    pysptk.sptk.uels
    pysptk.sptk.c2ir
    pysptk.sptk.lpc2c

    """

    return _sptk.c2acr(c, order, fftlen)


def c2ir(c, length=256):
    """Cepstrum to impulse response

    Parameters
    ----------
    c : array
         Cepstrum

    length : int, optional
         Length of impulse response. Default is 256.

    Returns
    -------
    h : array, shape (``length``)
        impulse response

    See Also
    --------
    pysptk.sptk.c2acr

    """

    return _sptk.c2ir(c, length)


def ic2ir(h, order=25):
    """Impulse response to cepstrum

    Parameters
    ----------
    h : array
         Impulse response

    order : int, optional
         Order of cepstrum. Default is 25.

    Returns
    -------
    c : array, shape (``order + 1``)
        Cepstrum

    See Also
    --------
    pysptk.sptk.c2ir

    """

    return _sptk.ic2ir(h, order)


def c2ndps(c, fftlen=256):
    """Cepstrum to Negative Derivative of Phase Spectrum (NDPS)

    Parameters
    ----------
    c : array
         Cepstrum

    fftlen : int, optional
         FFT length. Default is 256.

    Returns
    -------
    ndps : array, shape (``fftlen // 2 + 1``)
        NDPS

    Raises
    ------
    ValueError
        if non power of 2 ``fftlen`` is specified

    See Also
    --------
    pysptk.sptk.mgcep
    pysptk.sptk.ndps2c

    """

    return _sptk.c2ndps(c, fftlen)


def ndps2c(ndps, order=25):
    """Cepstrum to Negative Derivative of Phase Spectrum (NDPS)

    Parameters
    ----------
    ndps : array, shape (``fftlen // 2 + 1``)
        NDPS

    order : int, optional
        Order of cepstrum. Default is 25.

    Returns
    -------
    c : array, shape (``order + 1``)
         Cepstrum

    Raises
    ------
    ValueError
        if non power of 2 ``fftlen`` is detected

    See Also
    --------
    pysptk.sptk.mgc2sp
    pysptk.sptk.c2ndps

    """

    return _sptk.ndps2c(ndps, order)


def gc2gc(src_ceps, src_gamma=0.0, dst_order=None, dst_gamma=0.0):
    """Generalized cepstrum transform

    Parameters
    ----------
    src_ceps : array
        Generalized cepstrum.

    src_gamma : float, optional
        Gamma of source cepstrum. Default is 0.0.

    dst_order : int, optional
        Order of destination cepstrum. Default is ``len(src_ceps) - 1``.

    dst_gamma : float, optional
        Gamma of destination cepstrum. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape (``dst_order + 1``)
         Converted generalized cepstrum

    Raises
    ------
    ValueError
        - if invalid ``src_gamma`` is specified
        - if invalid ``dst_gamma`` is specified

    See Also
    --------
    pysptk.sptk.gcep
    pysptk.sptk.mgcep
    pysptk.sptk.freqt
    pysptk.sptk.mgc2mgc
    pysptk.sptk.lpc2c

    """

    return _sptk.gc2gc(src_ceps, src_gamma, dst_order, dst_gamma)


def gnorm(ceps, gamma=0.0):
    """Gain normalization

    Parameters
    ----------
    ceps : array
        Generalized cepstrum.

    gamma : float, optional
        Gamma. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape(same as ``ceps``)
        Normalized generalized cepstrum

    Raises
    ------
    ValueError
        if invalid ``gamma`` is specified

    See Also
    --------
    pysptk.sptk.ignorm
    pysptk.sptk.gcep
    pysptk.sptk.mgcep
    pysptk.sptk.gc2gc
    pysptk.sptk.mgc2mgc
    pysptk.sptk.freqt

    """

    return _sptk.gnorm(ceps, gamma)


def ignorm(ceps, gamma=0.0):
    """Inverse gain normalization

    Parameters
    ----------
    c : array
        Normalized generalized cepstrum

    gamma : float, optional
        Gamma. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape (same as ``ceps``)
        Generalized cepstrum

    Raises
    ------
    ValueError
        if invalid ``gamma`` is specified

    See Also
    --------
    pysptk.sptk.gnorm
    pysptk.sptk.gcep
    pysptk.sptk.mgcep
    pysptk.sptk.gc2gc
    pysptk.sptk.mgc2mgc
    pysptk.sptk.freqt

    """

    return _sptk.ignorm(ceps, gamma)


def freqt(ceps, order=25, alpha=0.0):
    """Frequency transform

    Parameters
    ----------
    ceps : array
        Cepstrum.

    order : int, optional
        Desired order of transformed cepstrum. Default is 25.

    alpha : float, optional
        All-pass constant. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape(``order + 1``)
        frequency transofmed cepsttrum (typically mel-cepstrum)

    See Also
    --------
    pysptk.sptk.mgc2mgc

    """

    return _sptk.freqt(ceps, order, alpha)


def frqtr(src_ceps, order=25, alpha=0.0):
    return _sptk.frqtr(src_ceps, order, alpha)


def mgc2mgc(src_ceps, src_alpha=0.0, src_gamma=0.0,
            dst_order=None, dst_alpha=0.0, dst_gamma=0.0):
    """Mel-generalized cepstrum transform

    Parameters
    ----------
    src_ceps : array
        Mel-generalized cepstrum.

    src_alpha : float, optional
        All-pass constant of source cesptrum. Default is 0.0.

    src_gamma : float, optional
        Gamma of source cepstrum. Default is 0.0.

    dst_order : int, optional
        Order of destination cepstrum. Default is ``len(src_ceps) - 1``.

    dst_alpha : float, optional
        All-pass constant of destination cesptrum. Default is 0.0.

    dst_gamma : float, optional
        Gamma of destination cepstrum. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape (``dst_order + 1``)
         Converted mel-generalized cepstrum

    Raises
    ------
    ValueError
        - if invalid ``src_gamma`` is specified
        - if invalid ``dst_gamma`` is specified

    See Also
    --------
    pysptk.sptk.uels
    pysptk.sptk.gcep
    pysptk.sptk.mcep
    pysptk.sptk.mgcep
    pysptk.sptk.gc2gc
    pysptk.sptk.freqt
    pysptk.sptk.lpc2c

    """

    return _sptk.mgc2mgc(src_ceps, src_alpha, src_gamma,
                         dst_order, dst_alpha, dst_gamma)


def mgc2sp(ceps, alpha=0.0, gamma=0.0, fftlen=256):
    """Mel-generalized cepstrum transform

    Parameters
    ----------
    ceps : array
        Mel-generalized cepstrum.

    alpha : float, optional
        All-pass constant. Default is 0.0.

    gamma : float, optional
        Gamma. Default is 0.0.

    fftlen : int, optional
        FFT length. Default is 256.

    Returns
    -------
    sp : array, shape (``fftlen // 2 + 1``)
         Complex spectrum

    Raises
    ------
    ValueError
        - if invalid ``gamma`` is specified
        - if non power of 2 ``fftlen`` is specified

    See Also
    --------
    pysptk.sptk.mgc2mgc
    pysptk.sptk.gc2gc
    pysptk.sptk.freqt
    pysptk.sptk.gnorm
    pysptk.sptk.lpc2c

    """

    return _sptk.mgc2sp(ceps, alpha, gamma, fftlen)


def mgclsp2sp(lsp, alpha=0.0, gamma=0.0, fftlen=256, gain=True):
    """MGC-LSP to spectrum

    Parameters
    ----------
    lsp : array
        MGC-LSP

    alpha : float, optional
        All-pass constant. Default is 0.0.

    gamma : float, optional
        Gamma. Default is 0.0.

    fftlen : int, optional
        FFT length. Default is 256.

    gain : bool, optional
        Whether the input MGC-LSP should have loggain or not.
        Default is True.

    Returns
    -------
    sp : array, shape (``fftlen // 2 + 1``)
         Complex spectrum

    Raises
    ------
    ValueError
        - if invalid ``gamma`` is specified
        - if non power of 2 ``fftlen`` is specified

    See Also
    --------
    pysptk.sptk.mgc2mgc

    """

    return _sptk.mgclsp2sp(lsp, alpha, gamma, fftlen, gain)

### F0 analysis ###


def swipe(x, fs, hopsize, min=60.0, max=240.0, threshold=0.3, otype="f0"):
    """SWIPE' - A Saw-tooth Waveform Inspired Pitch Estimation

    Parameters
    ----------
    x : array
        A whole audio signal

    fs : int
        Sampling frequency.

    hopsize : int
        Hop size.

    min : float, optional
        Minimum fundamental frequency. Default is 60.0

    max : float, optional
        Maximum fundamental frequency. Default is 240.0

    threshold : float, optional
        Voice/unvoiced threshold. Default is 0.3.

    otype : str or int, optional
        Output format
            (0) pitch
            (1) f0
            (2) log(f0)
        Default is f0.

    Returns
    -------
    f0  : array, shape(``np.ceil(float(len(x))/hopsize)``)
        Estimated f0 trajectory

    Raises
    ------
    ValueError
        if invalid otype is specified

    Examples
    --------

    >>> from scipy.io import wavfile
    >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
    >>> hopsize = 80 # 5ms for 16kHz data
    >>> f0 = pysptk.swipe(x.astype(np.float64), fs, 80, otype="f0")

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f0, linewidth=2, label="F0 trajectory estimated by SWIPE'")
    >>> plt.xlim(0, len(f0))
    >>> plt.legend()
    >>> plt.tight_layout()

    See Also
    --------
    pysptk.sptk.rapt

    """

    return _sptk.swipe(x, fs, hopsize, min, max, threshold, otype)


def rapt(x, fs, hopsize, min=60, max=240, voice_bias=0.0, otype="f0"):
    """RAPT - a robust algorithm for pitch tracking

    Parameters
    ----------
    x : array, dtype=np.float32
        A whole audio signal

    fs : int
        Sampling frequency.

    hopsize : int
        Hop size.

    min : float, optional
        Minimum fundamental frequency. Default is 60.0

    max : float, optional
        Maximum fundamental frequency. Default is 240.0

    voice_bias : float, optional
        Voice/unvoiced threshold. Default is 0.0.

    otype : str or int, optional
        Output format
            (0) pitch
            (1) f0
            (2) log(f0)
        Default is f0.

    Notes
    -----
    It is assumed that input array ``x`` has np.float32 dtype, while swipe
    assumes np.float64 dtype.

    Returns
    -------
    f0  : array, shape(``np.ceil(float(len(x))/hopsize)``)
        Estimated f0 trajectory

    Raises
    ------
    ValueError
        - if invalid min/max frequency specified
        - if invalid frame period specified (not in [1/fs, 0.1])
        - if input range too small for analysis by get_f0

    RuntimeError
        - problem in init_dp_f0()

    Please see also the RAPT code in SPTK for more detailed exception conditions.

    Examples
    --------

    >>> from scipy.io import wavfile
    >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
    >>> hopsize = 80 # 5ms for 16kHz data
    >>> f0 = pysptk.rapt(x.astype(np.float32), fs, 80, otype="f0")

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f0, linewidth=2, label="F0 trajectory estimated by RAPT")
    >>> plt.xlim(0, len(f0))
    >>> plt.legend()
    >>> plt.tight_layout()

    See Also
    --------
    pysptk.sptk.swipe

    """

    return _sptk.rapt(x, fs, hopsize, min, max, voice_bias, otype)

### Window functions ###


def blackman(n, normalize=1):
    """Blackman window

    Parameters
    ----------
    n : int
         Window length

    normalize : int, optional
        Normalization flag
            (0) don't normalize
            (1) normalize by power
            (2) normalize by magnitude

        Default is 1.

    Returns
    -------
    w : array, shape (n,)
        blackman window

    """

    return _sptk.blackman(n, normalize)


def hamming(n, normalize=1):
    """Hamming window

    Parameters
    ----------
    n : int
         Window length

    normalize : int, optional
        Normalization flag
            (0) don't normalize
            (1) normalize by power
            (2) normalize by magnitude

        Default is 1.

    Returns
    -------
    w : array, shape (n,)
        hamming window

    """

    return _sptk.hamming(n, normalize)


def hanning(n, normalize=1):
    """Hanning window

    Parameters
    ----------
    n : int
         Window length

    normalize : int, optional
        Normalization flag
            (0) don't normalize
            (1) normalize by power
            (2) normalize by magnitude

        Default is 1.

    Returns
    -------
    w : array, shape (n,)
        hanning window

    """

    return _sptk.hanning(n, normalize)


def bartlett(n, normalize=1):
    """Bartlett window

    Parameters
    ----------
    n : int
         Window length

    normalize : int, optional
        Normalization flag
            (0) don't normalize
            (1) normalize by power
            (2) normalize by magnitude

        Default is 1.

    Returns
    -------
    w : array, shape (n,)
        bartlett window

    """

    return _sptk.bartlett(n, normalize)


def trapezoid(n, normalize=1):
    """Trapezoid window

    Parameters
    ----------
    n : int
         Window length

    normalize : int, optional
        Normalization flag
            (0) don't normalize
            (1) normalize by power
            (2) normalize by magnitude

        Default is 1.

    Returns
    -------
    w : array, shape (n,)
        trapezoid window

    """

    return _sptk.trapezoid(n, normalize)


def rectangular(n, normalize=1):
    """Rectangular window

    Parameters
    ----------
    n : int
         Window length

    normalize : int, optional
        Normalization flag
            (0) don't normalize
            (1) normalize by power
            (2) normalize by magnitude

        Default is 1.

    Returns
    -------
    w : array, shape (n,)
        rectangular window

    """

    return _sptk.rectangular(n, normalize)


### Waveform generation filters ###


def poledf_delay(order):
    """Delay for poledf

    Parameters
    ----------
    order : int
        Order of poledf filter coefficients

    Returns
    -------
    delay : array
        Delay

    """
    return np.zeros(_sptk.poledf_delay_length(order))


def poledf(x, a, delay):
    """All-pole digital filter

    Parameters
    ----------
    x : float
        A input sample

    a : array
        AR coefficients

    delay : array
        Delay

    Returns
    -------
    y : float
        A filtered sample

    Raises
    ------
    ValueError
        if invalid delay length is supplied

    See Also
    --------
    pysptk.sptk.lpc
    pysptk.sptk.ltcdf
    pysptk.sptk.lmadf

    """

    return _sptk.poledf(x, a, delay)


def lmadf_delay(order, pd):
    """Delay for lmadf

    Parameters
    ----------
    order : int
        Order of lmadf filter coefficients

    pd : int
        Order of pade approximation.

    Returns
    -------
    delay : array
        Delay

    """

    return np.zeros(_sptk.lmadf_delay_length(order, pd))


def lmadf(x, b, pd, delay):
    """LMA digital filter

    Parameters
    ----------
    x : float
        A input sample

    c : array
        Cepstrum

    pd : int
        Order of pade approximation

    delay : array
        Delay

    Returns
    -------
    y : float
        A filtered sample

    Raises
    ------
    ValueError
        - if invalid order of pade approximation is specified
        - if invalid delay length is supplied

    See Also
    --------
    pysptk.sptk.uels
    pysptk.sptk.acep
    pysptk.sptk.poledf
    pysptk.sptk.ltcdf
    pysptk.sptk.glsadf
    pysptk.sptk.mlsadf
    pysptk.sptk.mglsadf

    """

    return _sptk.lmadf(x, b, pd, delay)


def lspdf_delay(order):
    """Delay for lspdf

    Parameters
    ----------
    order : int
        Order of lspdf filter coefficients

    Returns
    -------
    delay : array
        Delay

    """

    return np.zeros(_sptk.lspdf_delay_length(order))


def lspdf(x, f, delay):
    """LSP synthesis digital filter

    Parameters
    ----------
    x : float
        A input sample

    f : array
        LSP coefficients

    delay : array
        Delay

    Returns
    -------
    y : float
        A filtered sample

    Raises
    ------
    ValueError
        if invalid delay length is supplied

    See Also
    --------
    pysptk.sptk.lpc2lsp

    """

    return _sptk.lspdf(x, f, delay)


def ltcdf_delay(order):
    """Delay for ltcdf

    Parameters
    ----------
    order : int
        Order of ltcdf filter coefficients

    Returns
    -------
    delay : array
        Delay

    """

    return np.zeros(_sptk.ltcdf_delay_length(order))


def ltcdf(x, k, delay):
    """All-pole lattice digital filter

    Parameters
    ----------
    x : float
        A input sample

    k : array
        PARCOR coefficients.

    delay : array
        Delay

    Returns
    -------
    y : float
        A filtered sample

    Raises
    ------
    ValueError
        if invalid delay length is supplied

    See Also
    --------
    pysptk.sptk.lpc
    pysptk.sptk.lpc2par
    pysptk.sptk.lpc2lsp
    pysptk.sptk.poledf
    pysptk.sptk.lspdf

    """

    return _sptk.ltcdf(x, k, delay)


def glsadf_delay(order, stage):
    """Delay for glsadf

    Parameters
    ----------
    order : int
        Order of glsadf filter coefficients

    stage : int
        -1 / gamma

    Returns
    -------
    delay : array
        Delay

    """

    return np.zeros(_sptk.glsadf_delay_length(order, stage))


def glsadf(x, c,
           stage,
           delay):
    """GLSA digital filter

    Parameters
    ----------
    x : float
        A input sample

    c : array
        Geneeraized cepstrum

    stage : int
        -1 / gamma

    delay : array
        Delay

    Returns
    -------
    y : float
        A filtered sample

    Raises
    ------
    ValueError
        - if invalid number of stage is specified
        - if invalid delay length is supplied

    See Also
    --------
    pysptk.sptk.ltcdf
    pysptk.sptk.lmadf
    pysptk.sptk.lspdf
    pysptk.sptk.mlsadf
    pysptk.sptk.mglsadf

    """

    return _sptk.glsadf(x, c, stage, delay)


def mlsadf_delay(order, pd):
    """Delay for mlsadf

    Parameters
    ----------
    order : int
        Order of mlsadf filter coefficients

    pd : int
        Order of pade approximation.

    Returns
    -------
    delay : array
        Delay

    """

    return np.zeros(_sptk.mlsadf_delay_length(order, pd))


def mlsadf(x, b, alpha, pd, delay):
    """MLSA digital filter

    Parameters
    ----------
    x : float
        A input sample

    b : array
        MLSA filter coefficients

    alpha : float
        All-pass constant

    pd : int
        Order of pade approximation

    delay : array
        Delay

    Returns
    -------
    y : float
        A filtered sample

    Raises
    ------
    ValueError
        - if invalid order of pade approximation is specified
        - if invalid delay length is supplied

    See Also
    --------
    pysptk.sptk.mcep
    pysptk.sptk.amcep
    pysptk.sptk.poledf
    pysptk.sptk.ltcdf
    pysptk.sptk.lmadf
    pysptk.sptk.lspdf
    pysptk.sptk.glsadf
    pysptk.sptk.mglsadf

    """

    return _sptk.mlsadf(x, b, alpha, pd, delay)


def mglsadf_delay(order, stage):
    """Delay for mglsadf

    Parameters
    ----------
    order : int
        Order of mglsadf filter coefficients

    stage : int
        -1 / gamma

    Returns
    -------
    delay : array
        Delay

    """

    return np.zeros(_sptk.mglsadf_delay_length(order, stage))


def mglsadf(x, b, alpha, stage, delay):
    """MGLSA digital filter

    Parameters
    ----------
    x : float
        A input sample

    b : array
        MGLSA filter coefficients

    alpha : float
        All-pass constant

    stage : int
        -1 / gamma

    delay : array
        Delay

    Returns
    -------
    y : float
        A filtered sample

    Raises
    ------
    ValueError
        - if invalid number of stage is specified
        - if invalid delay length is supplied

    See Also
    --------
    pysptk.sptk.mgcep
    pysptk.sptk.poledf
    pysptk.sptk.ltcdf
    pysptk.sptk.lmadf
    pysptk.sptk.lspdf
    pysptk.sptk.mlsadf
    pysptk.sptk.glsadf

    """

    return _sptk.mglsadf(x, b, alpha, stage, delay)


### Excitation ###

def excite(pitch, hopsize=100, interp_period=1, gaussian=False, seed=1):
    """Excitation generation

    Parameters
    ----------
    pitch : array
        Pitch sequence.

        .. note::

            ``excite`` assumes that input is a **pitch** sequence, not **f0**
            sequence. Pitch sequence can be obtained by speficying
            ```otype="pitch"`` to F0 estimation methods.

    hopsize : int
        Hop size (frame period in sample). Default is 100.

    interp_period : int
        Interpolation period. Default is 1.

    gaussian : bool
        If True, generate gausssian noise for unvoiced frames, otherwise
        generate M-sequence. Default is False.

    seed : int
        Seed for nrand for Gaussian noise. Default is 1.

    Returns
    -------
    excitation : array
        Excitation signal

    See also
    --------

    pysptk.sptk.poledf
    pysptk.sptk.swipe
    pysptk.sptk.rapt


    """
    return _sptk.excite(pitch, hopsize, interp_period, gaussian, seed)


### Utils ###

def phidf(x, order, alpha, delay):
    _sptk.phidf(x, order, alpha, delay)


def lspcheck(lsp):
    return _sptk.lspcheck(lsp)
