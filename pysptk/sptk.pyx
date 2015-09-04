# coding: utf-8
# cython: boundscheck=True, wraparound=True

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
    lpc

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
cimport numpy as np

cimport cython
cimport sptk

from warnings import warn
from pysptk import assert_gamma, assert_fftlen, assert_pade


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
    return _agexp(r, x, y)


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
    return _gexp(r, x)


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
    return _glog(r, x)


def mseq():
    """M-sequence

    Returns
    -------
    A sample of m-sequence

    """
    return _mseq()


### Adaptive mel-generalized cepstrum analysis ###

def acep(x, np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
         lambda_coef=0.98, step=0.1, tau=0.9, pd=4, eps=1.0e-6):
    """Adaptive cepstral analysis

    Parameters
    ----------
    x : double
        A input sample

    c : array, shape(`order` + 1)
        Cepstrum. The result is stored in place.

    lambda_coef : float
        Leakage factor. Default is 0.98.

    step : float
        Step size. Default is 0.1.

    tau : float
        Momentum constant. Default is 0.9.

    pd : int
        Order of pade approximation. Default is 4.

    eps : float
        Minimum value for epsilon. Default is 1.0e-6.

    Returns
    -------
    prederr : float
        Prediction error

    Raises
    ------
    ValueError
        if invalid order of pade approximation is specified

    """
    assert_pade(pd)
    cdef int order = len(c) - 1
    cdef double prederr
    prederr = _acep(x, &c[0], order, lambda_coef, step, tau, pd, eps)
    return prederr


def agcep(x, np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
          stage=1,
          lambda_coef=0.98, step=0.1, tau=0.9, eps=1.0e-6):
    """Adaptive generalized cepstral analysis

    Parameters
    ----------
    x : float
        A input sample

    c : array, shape(`order` + 1)
        Cepstrum. The result is stored in-place.

    stage: int
        -1 / gamma. Default is 1.

    lambda_coef : float
        Leakage factor. Default is 0.98.

    step : float
        Step size. Default is 0.1.

    tau : float
        Momentum constant. Default is 0.9.

    eps : float
        Minimum value for epsilon. Default is 1.0e-6.

    Returns
    -------
    prederr : float
        Prediction error

    Raises
    ------
    ValueError
        if invalid number of stage is specified

    """
    if stage < 1:
        raise ValueError("stage >= 1 (-1 <= gamma < 0)")
    cdef int order = len(c) - 1
    cdef double prederr
    prederr = _agcep(x, &c[0], order, stage, lambda_coef, step, tau, eps)
    return prederr


def amcep(x, np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
          alpha=0.35,
          lambda_coef=0.98, step=0.1, tau=0.9, pd=4, eps=1.0e-6):
    """Adaptive mel-cepstral analysis

    Parameters
    ----------
    x : float
        A input sample

    b : array, shape(`order` + 1)
        MLSA filter coefficients. The result is stored in-place.

    alpha: float
        All-pass constant. Default is 0.35.

    lambda_coef : float
        Leakage factor. Default is 0.98.

    step : float
        Step size. Default is 0.1.

    tau : float
        Momentum constant. Default is 0.9.

    pd : int
        Order of pade approximation. Default is 4.

    eps : float
        Minimum value for epsilon. Default is 1.0e-6.

    Returns
    -------
    prederr : float
        Prediction error

    Raises
    ------
    ValueError
        if invalid order of pade approximation is specified

    """
    assert_pade(pd)
    cdef int order = len(b) - 1
    cdef double prederr
    prederr = _amcep(x, &b[0], order, alpha, lambda_coef, step, tau, pd, eps)
    return prederr


### Mel-generalized cepstrum analysis ###

def mcep(np.ndarray[np.float64_t, ndim=1, mode="c"] windowed not None,
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
    windowed : array, shape (`frame_len`)
        A windowed frame

    order : int
        Order of mel-cepstrum. Default is 25.

    alpha : float
        All pass constant. Default is 0.35.

    miniter : int
        Minimum number of iteration. Default is 2.

    maxiter : int
        Maximum number of iteration. Default is 30.

    threshold : float
        Threshold in theq. Default is 0.001.

    etype : int
        Type of parameter `eps`
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

        Default is 0.

    eps : float
        Initial value for log-periodogram or floor of periodogram in db.
        Default is 0.0.

    min_det : float
        Mimimum value of the determinant of normal matrix.
        Default is 1.0e-6

    itype : float
        Input data type:
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

        Default is 0.

    Returns
    -------
    mc : array, shape (`order + 1`)
        Mel-cepstrum

    Raises
    ------
    ValueError
        - if invalid `itype` is specified
        - if invalid `etype` is specified
        - if nonzero `eps` is specified when etype = 0
        - if negative `eps` is specified
        - if negative `min_det` is specified

    RuntimeError
        - if zero(s) are found in periodogram
        - if error happened in theq

    """

    if not itype in range(0, 5):
        raise ValueError("unsupported itype: %d, must be in 0:4" % itype)

    if not etype in range(0, 3):
        raise ValueError("unsupported etype: %d, must be in 0:2" % etype)

    if etype == 0 and eps != 0.0:
        raise ValueError("eps cannot be specified for etype = 0")

    if (etype == 1 or etype == 2) and eps < 0.0:
        raise ValueError("eps: %f, must be >= 0" % eps)

    if min_det < 0.0:
        raise ValueError("min_det must be positive: min_det = %f" % min_det)

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] mc
    cdef int windowed_length = len(windowed)
    cdef int ret
    mc = np.zeros(order + 1, dtype=np.float64)
    ret = _mcep(&windowed[0], windowed_length, &mc[0],
                order, alpha, miniter, maxiter, threshold, etype, eps,
                min_det, itype)
    assert ret == -1 or ret == 0 or ret == 3 or ret == 4
    if ret == 3:
        raise RuntimeError("failed to compute mcep; error occured in theq")
    elif ret == 4:
        raise RuntimeError(
            "zero(s) are found in periodogram, use eps option to floor")

    return mc


def gcep(np.ndarray[np.float64_t, ndim=1, mode="c"] windowed not None,
         order=25, gamma=0.0,
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
    windowed : array, shape (`frame_len`)
        A windowed frame

    order : int
        Order of generalized-cepstrum. Default is 25.

    gamma : float
        Parameter of generalized log function. Default is 0.0.

    miniter : int
        Minimum number of iteration. Default is 2.

    maxiter : int
        Maximum number of iteration. Default is 30.

    threshold : float
        Threshold in theq. Default is 0.001

    etype : int
        Type of parameter `eps`
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

        Default is 0.

    eps : float
        Initial value for log-periodogram or floor of periodogram in db.
        Default is 0.0.

    min_det : float
        Mimimum value of the determinant of normal matrix. Default is 1.0e-6.

    itype : float
        Input data type:
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

        Default is 0.

    Returns
    -------
    gc : array, shape (`order + 1`)
        Generalized cepstrum

    Raises
    ------
    ValueError
        - if invalid `itype` is specified
        - if invalid `etype` is specified
        - if nonzero `eps` is specified when etype = 0
        - if negative `eps` is specified
        - if negative `min_det` is specified

    RuntimeError
        - if error happened in theq

    """

    assert_gamma(gamma)
    if not itype in range(0, 5):
        raise ValueError("unsupported itype: %d, must be in 0:4" % itype)

    if not etype in range(0, 3):
        raise ValueError("unsupported etype: %d, must be in 0:2" % etype)

    if etype == 0 and eps != 0.0:
        raise ValueError("eps cannot be specified for etype = 0")

    if (etype == 1 or etype == 2) and eps < 0.0:
        raise ValueError("eps: %f, must be >= 0" % eps)

    if min_det < 0.0:
        raise ValueError("min_det must be positive: min_det = %f" % min_det)

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] gc
    cdef int windowed_length = len(windowed)
    cdef int ret
    gc = np.zeros(order + 1, dtype=np.float64)
    ret = _gcep(&windowed[0], windowed_length, &gc[0], order,
                gamma, miniter, maxiter, threshold, etype, eps, min_det, itype)
    assert ret == -1 or ret == 0 or ret == 3
    if ret == 3:
        raise RuntimeError("failed to compute gcep; error occured in theq")

    if not norm:
        _ignorm(&gc[0], &gc[0], order, gamma)

    return gc


@cython.boundscheck(False)
@cython.wraparound(False)
def mgcep(np.ndarray[np.float64_t, ndim=1, mode="c"] windowed not None,
          order=25, alpha=0.35, gamma=0.0,
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
    windowed : array, shape (`frame_len`)
        A windowed frame

    order : int
        Order of mel-generalized cepstrum. Default is 25.

    alpha : float
        All pass constant. Default is 0.35.

    gamma : float
        Parameter of generalized log function. Default is 0.0.

    num_recursions : int
        Number of recursions. Default is `len(windowed)` - 1.

    miniter : int
        Minimum number of iteration. Default is 2.

    maxiter : int
        Maximum number of iteration. Default is 30.

    threshold : float
        Threshold. Default is 0.001.

    etype : int
        Type of paramter `e`
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

        Default is 0.

    eps : float
        Initial value for log-periodogram or floor of periodogram in db.
        Default is 0.0.

    min_det : float
        Mimimum value of the determinant of normal matrix.
        Default is 1.0e-6.

    itype : float
        Input data type:
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

        Default is 0.

    otype : int
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
    mgc : array, shape (`order + 1`)
        mel-generalized cepstrum

    Raises
    ------
    ValueError
        - if invalid `itype` is specified
        - if invalid `etype` is specified
        - if nonzero `eps` is specified when etype = 0
        - if negative `eps` is specified
        - if negative `min_det` is specified
        - if invalid `otype` is specified

    RuntimeError
        - if error happened in theq

    """

    assert_gamma(gamma)
    if not itype in range(0, 5):
        raise ValueError("unsupported itype: %d, must be in 0:4" % itype)

    if not etype in range(0, 3):
        raise ValueError("unsupported etype: %d, must be in 0:2" % etype)

    if etype == 0 and eps != 0.0:
        raise ValueError("eps cannot be specified for etype = 0")

    if (etype == 1 or etype == 2) and eps < 0.0:
        raise ValueError("eps: %f, must be >= 0" % eps)

    if min_det < 0.0:
        raise ValueError("min_det must be positive: min_det = %f" % min_det)

    if not otype in range(0, 6):
        raise ValueError("unsupported otype: %d, must be in 0:5" % otype)

    if num_recursions is None:
        num_recursions = len(windowed) - 1

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] mgc
    cdef int windowed_length = len(windowed)
    cdef int ret
    mgc = np.zeros(order + 1, dtype=np.float64)
    ret = _mgcep(&windowed[0], windowed_length, &mgc[0],
                 order, alpha, gamma, num_recursions, miniter, maxiter,
                 threshold, etype, eps, min_det, itype)
    assert ret == -1 or ret == 0 or ret == 3
    if ret == 3:
        raise RuntimeError("failed to compute mgcep; error occured in theq")

    if otype == 0 or otype == 1 or otype == 2 or otype == 4:
        _ignorm(&mgc[0], &mgc[0], order, gamma)

    if otype == 0 or otype == 2 or otype == 4:
        _b2mc(&mgc[0], &mgc[0], order, alpha)

    if otype == 2 or otype == 4:
        _gnorm(&mgc[0], &mgc[0], order, gamma)

    cdef int i = 0
    cdef double g = gamma
    if otype == 4 or otype == 5:
        for i in range(1, len(mgc)):
            mgc[i] *= g

    return mgc


def uels(np.ndarray[np.float64_t, ndim=1, mode="c"] windowed not None,
         order=25,
         miniter=2,
         maxiter=30,
         threshold=0.001,
         etype=0,
         eps=0.0,
         itype=0):
    """Unbiased estimation of log spectrum

    Parameters
    ----------
    windowed : array, shape (`frame_len`)
        A windowed frame

    order : int
        Order of cepstrum. Default is 25.

    miniter : int
        Minimum number of iteration. Default is 2.

    maxiter : int
        Maximum number of iteration. Default is 30.

    threshold : float
        Threshold in theq. Default is 0.001

    etype : int
        Type of parameter `eps`
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

        Default is 0.

    eps : float
        Initial value for log-periodogram or floor of periodogram in db.
        Default is 0.0.

    itype : float
        Input data type:
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

        Default is 0.

    Returns
    -------
    c : array, shape (`order + 1`)
        cepstrum estimated by uels

    Raises
    ------
    ValueError
        - if invalid `itype` is specified
        - if invalid `etype` is specified
        - if nonzero `eps` is specified when etype = 0
        - if negative `eps` is specified

    RuntimeError
        - if zero(s) are found in periodogram

    """

    if not itype in range(0, 5):
        raise ValueError("unsupported itype: %d, must be in 0:4" % itype)

    if not etype in range(0, 3):
        raise ValueError("unsupported etype: %d, must be in 0:2" % etype)

    if etype == 0 and eps != 0.0:
        raise ValueError("eps cannot be specified for etype = 0")

    if (etype == 1 or etype == 2) and eps < 0.0:
        raise ValueError("eps: %f, must be >= 0" % eps)

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] c
    cdef int windowed_length = len(windowed)
    cdef int ret
    c = np.zeros(order + 1, dtype=np.float64)
    ret = _uels(&windowed[0], windowed_length, &c[0], order,
                miniter, maxiter, threshold, etype, eps, itype)
    assert ret == -1 or ret == 0 or ret == 3
    if ret == 3:
        raise RuntimeError(
            "zero(s) are found in periodogram, use eps option to floor")

    return c


def fftcep(np.ndarray[np.float64_t, ndim=1, mode="c"] logsp not None,
           order=25,
           num_iter=0,
           acceleration_factor=0.0):
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] c
    cdef int logsp_length = len(logsp)
    c = np.zeros(order + 1, dtype=np.float64)
    _fftcep(&logsp[0], logsp_length, &c[0], order + 1,
            num_iter, acceleration_factor)

    return c


def lpc(np.ndarray[np.float64_t, ndim=1, mode="c"] windowed not None,
        order=25,
        min_det=1.0e-6):
    """Linear prediction analysis

    Parameters
    ----------
    windowed : array, shape (`frame_len`)
        A windowed frame

    order : int
        Order of LPC. Default is 25.

    min_det : float
        Mimimum value of the determinant of normal matrix.
        Default is 1.0e-6.

    Returns
    -------
    a : array, shape (`order + 1`)
        LPC

    Raises
    ------
    ValueError
        - if negative `min_det` is specified

    RuntimeError
        - if error happened in levdur

    """

    if min_det < 0.0:
        raise ValueError("min_det must be positive: min_det = %f" % min_det)

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] a
    cdef int windowed_length = len(windowed)
    cdef int ret
    a = np.zeros(order + 1, dtype=np.float64)
    ret = _lpc(&windowed[0], windowed_length, &a[0], order, min_det)
    assert ret == -2 or ret == -1 or ret == 0
    if ret == -2:
        warn("failed to compute `stable` LPC. Please try again with different paramters")
    elif ret == -1:
        raise RuntimeError(
            "failed to compute LPC. Please try again with different parameters")

    return a


### LPC, LSP and PARCOR conversions ###

def lpc2c(np.ndarray[np.float64_t, ndim=1, mode="c"] lpc not None,
          order=None):
    """LPC to cepstrum

    Parameters
    ----------
    lpc : array
        LPC

    order : int
        Order of cepstrum. Default is `len(lpc)` - 1.

    Returns
    -------
    ceps : array, shape (`order + 1`)
        cepstrum

    """

    if order is None:
        order = len(lpc) - 1

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] ceps
    cdef int src_order = len(lpc) - 1
    ceps = np.zeros(order + 1, dtype=np.float64)
    _lpc2c(&lpc[0], src_order, &ceps[0], order)
    return ceps


@cython.boundscheck(False)
@cython.wraparound(False)
def lpc2lsp(np.ndarray[np.float64_t, ndim=1, mode="c"] lpc not None,
            numsp=512, maxiter=4, eps=1.0e-6, loggain=False, otype=0,
            fs=None):
    """LPC to LSP

    Parameters
    ----------
    lpc : array
        LPC

    numsp : int
        Number of unit circle. Default is 512.

    maxiter : int
        Maximum number of iteration. Default is 4.

    eps : float
        End condition for iteration. Default is 1.0e-6.

    loggain : bool
        whether the converted lsp should have loggain or not.
        Default is False.

    fs : int
        Sampling frequency. Default is None and unused.

    otype : int
        Output format LSP
            0  normalized frequency (0 ~ pi)
            1  normalized frequency (0 ~ 0.5)
            2  frequency (kHz)
            3  frequency (Hz)

        Default is 0.

    Returns
    -------
    lsp : array, shape (`order + 1`)
        LSP

    raises
    ------
    ValueError
        if `fs` is not specified when otype = 2 or 3.

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] lsp
    cdef int order = len(lpc) - 1
    lsp = np.zeros_like(lpc)
    _lpc2lsp(&lpc[0], &lsp[0], order, numsp, maxiter, eps)

    if otype == 0:
        lsp[1:] *= 2 * np.pi
    elif otype == 2 or otype == 3:
        if fs is None:
            raise ValueError("fs must be specified when otype == 2 or 3")
        lsp[1:] *= fs

    if otype == 3:
        lsp[1:] *= 1000.0

    if loggain:
        lsp[0] = np.log(lpc[0])
    else:
        lsp[0] = lpc[0]

    return lsp


def lpc2par(np.ndarray[np.float64_t, ndim=1, mode="c"] lpc not None):
    """LPC to PARCOR

    Parameters
    ----------
    lpc : array
        LPC

    Returns
    -------
    par : array, shape (same as `lpc`)
        PARCOR

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] par
    par = np.zeros_like(lpc)
    cdef int order = len(lpc) - 1
    _lpc2par(&lpc[0], &par[0], order)
    return par


def par2lpc(np.ndarray[np.float64_t, ndim=1, mode="c"] par not None):
    """PARCOR to LPC

    Parameters
    ----------
    par : array
        PARCOR

    Returns
    -------
    lpc : array, shape (same as `par`)
        LPC

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] lpc
    lpc = np.zeros_like(par)
    cdef int order = len(par) - 1
    _par2lpc(&par[0], &lpc[0], order)
    return lpc


def lsp2sp(np.ndarray[np.float64_t, ndim=1, mode="c"] lsp not None,
           fftlen=256):
    """LSP to spectrum

    Parameters
    ----------
    lsp : array
        LSP

    TODO: consider `otype` optional argument

    Returns
    -------
    sp : array, shape
        Spectrum. ln|H(z)|.

    Notes
    -----
    It is asuumed that `lsp` has loggain at `lsp[0]`.

    """

    assert_fftlen(fftlen)
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] sp
    cdef int sp_length = (fftlen >> 1) + 1
    sp = np.zeros(sp_length, dtype=np.float64)
    cdef int order = len(lsp) - 1
    _lsp2sp(&lsp[0], order, &sp[0], sp_length, 1)
    return sp


### Mel-generalized cepstrum conversions ###

def mc2b(np.ndarray[np.float64_t, ndim=1, mode="c"] mc not None,
         alpha=0.35):
    """Mel-cepsrum to MLSA filter coefficients

    Parameters
    ----------
    mc : array, shape
        Mel-cepstrum.

    alpha : float
        All-pass constant. Default is 0.35.

    Returns
    -------
    b : array, shape(same as `mc`)
        MLSA filter coefficients

    """
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] b
    b = np.zeros_like(mc)
    cdef int order = len(mc) - 1
    _mc2b(&mc[0], &b[0], order, alpha)
    return b


def b2mc(np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
         alpha=0.35):
    """MLSA filter coefficients to mel-cesptrum

    Parameters
    ----------
    b : array, shape
        MLSA filter coefficients

    alpha : float
        All-pass constant. Default is 0.35.

    Returns
    -------
    mc : array, shape (same as `b`)
        Mel-cepstrum.

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] mc
    mc = np.zeros_like(b)
    cdef int order = len(b) - 1
    _b2mc(&b[0], &mc[0], order, alpha)
    return mc


def b2c(np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
        dst_order=None,
        alpha=0.35):
    cdef int src_order = len(b) - 1
    if dst_order is None:
        dst_order = src_order
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] c
    c = np.zeros(dst_order + 1, dtype=np.float64)
    _b2c(&b[0], src_order, &c[0], dst_order, alpha)
    return c


def c2acr(np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
          order=None,
          fftlen=256):
    """Cepstrum to autocorrelation

    Parameters
    ----------
    c : array
        Cepstrum

    order : int
        Order of cepstrum. Default is `len(c) - 1`.

    fftlen : int
        FFT length. Default is 256.

    Returns
    -------
    r : array, shape (`order` + 1)
        Autocorrelation

    Raises
    ------
    ValueError
        if non power of 2 `fftlen` is specified

    """

    assert_fftlen(fftlen)
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] r
    cdef int src_order = len(c) - 1
    if order is None:
        order = src_order
    r = np.zeros(order + 1, dtype=np.float64)
    _c2acr(&c[0], src_order, &r[0], order, fftlen)
    return r


def c2ir(np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
         length=256):
    """Cepstrum to impulse response

    Parameters
    ----------
    c : array
         Cepstrum

    length : int
         Length of impulse response. Default is 256.

    Returns
    -------
    h : array, shape (`length`)
        impulse response

    """

    cdef int order = len(c)  # NOT len(c) - 1
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] h
    h = np.zeros(length, dtype=np.float64)
    _c2ir(&c[0], order, &h[0], length)
    return h


def ic2ir(np.ndarray[np.float64_t, ndim=1, mode="c"] h not None,
          order=25):
    """Impulse response to cepstrum

    Parameters
    ----------
    h : array
         Impulse response

    order : int
         Order of cepstrum. Default is 25.

    Returns
    -------
    c : array, shape (`order` + 1)
        Cepstrum

    """

    cdef int length = len(h)
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] c
    c = np.zeros(order + 1, dtype=np.float64)
    _ic2ir(&h[0], length, &c[0], len(c))
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
def c2ndps(np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
           fftlen=256):
    """Cepstrum to Negative Derivative of Phase Spectrum (NDPS)

    Parameters
    ----------
    c : array
         Cepstrum

    fftlen : int
         FFT length. Default is 256.

    Returns
    -------
    ndps : array, shape (`fftlen // 2 + 1`)
        NDPS

    Raises
    ------
    ValueError
        if non power of 2 `fftlen` is specified

    """
    assert_fftlen(fftlen)
    cdef int dst_length = (fftlen >> 1) + 1
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] ndps, buf
    ndps = np.zeros(dst_length, dtype=np.float64)
    cdef int order = len(c) - 1
    buf = np.zeros(fftlen, dtype=np.float64)
    _c2ndps(&c[0], order, &buf[0], fftlen)

    buf[0:dst_length] = buf[0:dst_length]

    return ndps


def ndps2c(np.ndarray[np.float64_t, ndim=1, mode="c"] ndps not None,
           order=25):
    """Cepstrum to Negative Derivative of Phase Spectrum (NDPS)

    Parameters
    ----------
    ndps : array, shape (`fftlen` // 2 + 1)
        NDPS

    order : int
        Order of cepstrum. Default is 25.

    Returns
    -------
    c : array, shape (`order` + 1)
         Cepstrum

    Raises
    ------
    ValueError
        if non power of 2 `fftlen` is detected

    """

    # assuming the lenght of ndps is fftlen/2+1
    cdef int fftlen = (len(ndps) - 1) << 1
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] c
    assert_fftlen(fftlen)
    c = np.zeros(order + 1, dtype=np.float64)
    _ndps2c(&ndps[0], fftlen, &c[0], order)
    return ndps


def gc2gc(np.ndarray[np.float64_t, ndim=1, mode="c"] src_ceps not None,
          src_gamma=0.0, dst_order=None, dst_gamma=0.0):
    """Generalized cepstrum transform

    Parameters
    ----------
    src_ceps : array
        Generalized cepstrum.

    src_gamma : float
        Gamma of source cepstrum. Default is 0.0.

    dst_order : int
        Order of destination cepstrum. Default is `len(src_ceps) - 1`.

    dst_gamma : float
        Gamma of destination cepstrum. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape (`dst_order` + 1)
         Converted generalized cepstrum

    Raises
    ------
    ValueError
        - if invalid `src_gamma` is specified
        - if invalid `dst_gamma` is specified

    """

    assert_gamma(src_gamma)
    assert_gamma(dst_gamma)

    cdef int src_order = len(src_ceps) - 1
    if dst_order is None:
        dst_order = src_order
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_ceps
    dst_ceps = np.zeros(dst_order + 1, dtype=np.float64)

    _gc2gc(&src_ceps[0], src_order, src_gamma,
           &dst_ceps[0], dst_order, dst_gamma)

    return dst_ceps


def gnorm(np.ndarray[np.float64_t, ndim=1, mode="c"] ceps not None,
          gamma=0.0):
    """Gain normalization

    Parameters
    ----------
    ceps : array, shape
        Generalized cepstrum.

    gamma : float
        Gamma. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape(same as `ceps`)
        Normalized generalized cepstrum

    Raises
    ------
    ValueError
        if invalid `gamma` is specified

    """

    assert_gamma(gamma)
    cdef int order = len(ceps) - 1
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_ceps
    dst_ceps = np.zeros_like(ceps)
    _gnorm(&ceps[0], &dst_ceps[0], order, gamma)
    return dst_ceps


def ignorm(np.ndarray[np.float64_t, ndim=1, mode="c"] ceps not None,
           gamma=0.0):
    """Inverse gain normalization

    Parameters
    ----------
    c : array
        Normalized generalized cepstrum

    gamma : float
        Gamma. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape (same as `ceps`)
        Generalized cepstrum

    Raises
    ------
    ValueError
        if invalid `gamma` is specified

    """

    assert_gamma(gamma)
    cdef int order = len(ceps) - 1
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_ceps
    dst_ceps = np.zeros_like(ceps)
    _ignorm(&ceps[0], &dst_ceps[0], order, gamma)
    return dst_ceps


def freqt(np.ndarray[np.float64_t, ndim=1, mode="c"] ceps not None,
          order=25, alpha=0.0):
    """Frequency transform

    Parameters
    ----------
    ceps : array
        Cepstrum.

    order : int
        Desired order of transformed cepstrum. Default is 25.

    alpha : float
        All-pass constant. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape(`order` + 1)
        frequency transofmed cepsttrum (typically mel-cepstrum)

    """

    cdef int src_order = len(ceps) - 1
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_ceps
    dst_ceps = np.zeros(order + 1, dtype=np.float64)
    _freqt(&ceps[0], src_order, &dst_ceps[0], order, alpha)
    return dst_ceps


def frqtr(np.ndarray[np.float64_t, ndim=1, mode="c"] src_ceps not None,
          order=25, alpha=0.0):
    cdef int src_order = len(src_ceps) - 1
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_ceps
    dst_ceps = np.zeros(order + 1, dtype=np.float64)
    _frqtr(&src_ceps[0], src_order, &dst_ceps[0], order, alpha)
    return dst_ceps


def mgc2mgc(np.ndarray[np.float64_t, ndim=1, mode="c"] src_ceps not None,
            src_alpha=0.0, src_gamma=0.0,
            dst_order=None, dst_alpha=0.0, dst_gamma=0.0):
    """Mel-generalized cepstrum transform

    Parameters
    ----------
    src_ceps : array
        Mel-generalized cepstrum.

    src_alpha : float
        All-pass constant of source cesptrum. Default is 0.0.

    src_gamma : float
        Gamma of source cepstrum. Default is 0.0.

    dst_order : int
        Order of destination cepstrum. Default is `len(src_ceps) - 1`.

    dst_alpha : float
        All-pass constant of destination cesptrum. Default is 0.0.

    dst_gamma : float
        Gamma of destination cepstrum. Default is 0.0.

    Returns
    -------
    dst_ceps : array, shape (`dst_order` + 1)
         Converted mel-generalized cepstrum

    Raises
    ------
    ValueError
        - if invalid `src_gamma` is specified
        - if invalid `dst_gamma` is specified

    """

    assert_gamma(src_gamma)
    assert_gamma(dst_gamma)

    cdef int src_order = len(src_ceps) - 1
    if dst_order is None:
        dst_order = src_order
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_ceps
    dst_ceps = np.zeros(dst_order + 1, dtype=np.float64)

    _mgc2mgc(&src_ceps[0], src_order, src_alpha, src_gamma,
              &dst_ceps[0], dst_order, dst_alpha, dst_gamma)

    return dst_ceps


@cython.boundscheck(False)
@cython.wraparound(False)
def mgc2sp(np.ndarray[np.float64_t, ndim=1, mode="c"] ceps not None,
           alpha=0.0, gamma=0.0, fftlen=256):
    """Mel-generalized cepstrum transform

    Parameters
    ----------
    ceps : array
        Mel-generalized cepstrum.

    alpha : float
        All-pass constant. Default is 0.0.

    gamma : float
        Gamma. Default is 0.0.

    fftlen : int
        FFT length. Default is 256.

    Returns
    -------
    sp : array, shape (`fftlen` // 2 + 1)
         Complex spectrum

    Raises
    ------
    ValueError
        - if invalid `gamma` is specified
        - if non power of 2 `fftlen` is specified

    """
    assert_gamma(gamma)
    assert_fftlen(fftlen)

    cdef int order = len(ceps) - 1
    cdef np.ndarray[np.complex128_t, ndim = 1, mode = "c"] sp
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] sp_r, sp_i

    sp = np.zeros((fftlen >> 1) + 1, dtype=np.complex128)
    sp_r = np.zeros(fftlen, dtype=np.float64)
    sp_i = np.zeros(fftlen, dtype=np.float64)

    _mgc2sp(&ceps[0], order, alpha, gamma, &sp_r[0], &sp_i[0], fftlen)

    cdef int i
    for i in range(0, len(sp)):
        sp[i] = sp_r[i] + sp_i[i] * 1j

    return sp


def mgclsp2sp(np.ndarray[np.float64_t, ndim=1, mode="c"] lsp not None,
              alpha=0.0, gamma=0.0, fftlen=256, gain=True):
    """MGC-LSP to spectrum

    Parameters
    ----------
    lsp : array
        MGC-LSP

    alpha : float
        All-pass constant. Default is 0.0.

    gamma : float
        Gamma. Default is 0.0.

    fftlen : int
        FFT length. Default is 256.

    gain : bool
        Whether the input MGC-LSP should have loggain or not.
        Default is True.

    Returns
    -------
    sp : array, shape (`fftlen` // 2 + 1)
         Complex spectrum

    Raises
    ------
    ValueError
        - if invalid `gamma` is specified
        - if non power of 2 `fftlen` is specified

    """
    assert_gamma(gamma)
    assert_fftlen(fftlen)

    cdef int order = gain if len(lsp) - 1 else len(lsp)
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] sp
    sp = np.zeros((fftlen >> 1) + 1, dtype=np.float64)

    _mgclsp2sp(alpha, gamma, &lsp[0], order, &sp[0], len(sp), int(gain))

    return sp


### F0 analysis ###

def swipe(np.ndarray[np.float64_t, ndim=1, mode="c"] x not None,
          fs, hopsize,
          min=50.0, max=800.0, threshold=0.3, otype=1):
    """SWIPE' - A Saw-tooth Waveform Inspired Pitch Estimation

    Parameters
    ----------
    x : array, shape
        A whole audio signal

    fs : int
        Sampling frequency.

    hopsize : int
        Hop size.

    min : float
        Minimum fundamental frequency. Default is 50.0

    max : float
        Maximum fundamental frequency. Default is 800.0

    threshold : float
        Voice/unvoiced threshold. Default is 0.3.

    otype : int (default=1)
        Output format (0) pitch (1) f0 (2) log(f0).

    Returns
    -------
    f0  : array, shape(`len(x)/frame_shift+1`)
        Estimated f0 trajectory

    Raises
    ------
    ValueError
        if invalid otype is specified

    """
    if not otype in range(0, 3):
        raise ValueError("otype must be 0, 1, or 2")

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] f0
    cdef int x_length = len(x)
    cdef int expected_len = int(x_length / hopsize) + 1

    f0 = np.zeros(expected_len, dtype=np.float64)

    _swipe(&x[0], &f0[0], x_length, fs, hopsize, min, max, threshold, otype)
    return f0


### Window functions ###

cdef __window(Window window_type, np.ndarray[np.float64_t, ndim=1, mode="c"] x,
              int size, int normalize):
    if normalize < 0 or normalize > 2:
        raise ValueError("normalize must be 0, 1 or 2")
    cdef double g = _window(window_type, &x[0], size, normalize)
    return x


def blackman(n, normalize=1):
    """Blackman window

    Parameters
    ----------
    n : int
         Window length

    normalize : int
        Normalization flag.
            0 : don't normalize
            1 : normalize by power
            2 : normalize by magnitude

        Defalt is 0.

    Returns
    -------
    w : array, shape (n,)
        blackman window

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] x
    x = np.ones(n, dtype=np.float64)
    cdef Window window_type = BLACKMAN
    return __window(window_type, x, len(x), normalize)


def hamming(n, normalize=1):
    """Hamming window

    Parameters
    ----------
    n : int
         Window length

    normalize : int
        Normalization flag.
            0 : don't normalize
            1 : normalize by power
            2 : normalize by magnitude

        Defalt is 0.

    Returns
    -------
    w : array, shape (n,)
        hamming window

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] x
    x = np.ones(n, dtype=np.float64)
    cdef Window window_type = HAMMING
    return __window(window_type, x, len(x), normalize)


def hanning(n, normalize=1):
    """Hanning window

    Parameters
    ----------
    n : int
         Window length

    normalize : int
        Normalization flag.
            0 : don't normalize
            1 : normalize by power
            2 : normalize by magnitude

        Defalt is 0.

    Returns
    -------
    w : array, shape (n,)
        hanning window

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] x
    x = np.ones(n, dtype=np.float64)
    cdef Window window_type = HANNING
    return __window(window_type, x, len(x), normalize)


def bartlett(n, normalize=1):
    """Bartlett window

    Parameters
    ----------
    n : int
         Window length

    normalize : int
        Normalization flag.
            0 : don't normalize
            1 : normalize by power
            2 : normalize by magnitude

        Defalt is 0.

    Returns
    -------
    w : array, shape (n,)
        bartlett window

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] x
    x = np.ones(n, dtype=np.float64)
    cdef Window window_type = BARTLETT
    return __window(window_type, x, len(x), normalize)


def trapezoid(n, normalize=1):
    """Trapezoid window

    Parameters
    ----------
    n : int
         Window length

    normalize : int
        Normalization flag.
            0 : don't normalize
            1 : normalize by power
            2 : normalize by magnitude

        Defalt is 0.

    Returns
    -------
    w : array, shape (n,)
        trapezoid window

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] x
    x = np.ones(n, dtype=np.float64)
    cdef Window window_type = TRAPEZOID
    return __window(window_type, x, len(x), normalize)


def rectangular(n, normalize=1):
    """Rectangular window

    Parameters
    ----------
    n : int
         Window length

    normalize : int
        Normalization flag.
            0 : don't normalize
            1 : normalize by power
            2 : normalize by magnitude

        Defalt is 0.

    Returns
    -------
    w : array, shape (n,)
        rectangular window

    """

    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] x
    x = np.ones(n, dtype=np.float64)
    cdef Window window_type = RECTANGULAR
    return __window(window_type, x, len(x), normalize)


### Waveform generation filters ###

def poledf_delay_length(order):
    return order


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
    return np.zeros(poledf_delay_length(order))


def poledf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] a not None,
           np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
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

    """

    cdef int order = len(a) - 1
    if len(delay) != poledf_delay_length(order):
        raise ValueError("inconsistent delay length")

    return _poledf(x, &a[0], order, &delay[0])


def lmadf_delay_length(order, pd):
    return 2 * pd * (order + 1)


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

    return np.zeros(lmadf_delay_length(order, pd))


def lmadf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
          pd,
          np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
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

    """
    assert_pade(pd)

    cdef int order = len(b) - 1
    if len(delay) != lmadf_delay_length(order, pd):
        raise ValueError("inconsistent delay length")

    return _lmadf(x, &b[0], order, pd, &delay[0])


def lspdf_delay_length(order):
    return 2 * order + 1


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

    return np.zeros(lspdf_delay_length(order))


def lspdf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] f not None,
          np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
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

    """

    cdef int order = len(f) - 1
    if len(delay) != lspdf_delay_length(order):
        raise ValueError("inconsistent delay length")

    if order % 2 == 0:
        return _lspdf_even(x, &f[0], order, &delay[0])
    else:
        return _lspdf_odd(x, &f[0], order, &delay[0])


def ltcdf_delay_length(order):
    return order + 1


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

    return np.zeros(ltcdf_delay_length(order))


def ltcdf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] k not None,
          np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
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

    """

    cdef int order = len(k) - 1
    if len(delay) != ltcdf_delay_length(order):
        raise ValueError("inconsistent delay length")

    return _ltcdf(x, &k[0], order, &delay[0])


def glsadf_delay_length(order, stage):
    return order * (stage + 1) + 1


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

    return np.zeros(glsadf_delay_length(order, stage))


def glsadf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
           stage,
           np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
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

    """

    if stage < 1:
        raise ValueError("stage >= 1 (-1 <= gamma < 0)")

    cdef int order = len(c) - 1
    if len(delay) != glsadf_delay_length(order, stage):
        raise ValueError("inconsistent delay length")

    return _glsadf(x, &c[0], order, stage, &delay[0])


def mlsadf_delay_length(order, pd):
    return 3 * (pd + 1) + pd * (order + 2)


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

    return np.zeros(mlsadf_delay_length(order, pd))


def mlsadf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
           alpha, pd,
           np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
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

    """

    assert_pade(pd)

    cdef int order = len(b) - 1
    if len(delay) != mlsadf_delay_length(order, pd):
        raise ValueError("inconsistent delay length")

    return _mlsadf(x, &b[0], order, alpha, pd, &delay[0])


def mglsadf_delay_length(order, stage):
    return (order + 1) * stage


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

    return np.zeros(mglsadf_delay_length(order, stage))


def mglsadf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
            alpha, stage,
            np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
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

    """

    if stage < 1:
        raise ValueError("stage >= 1 (-1 <= gamma < 0)")

    cdef int order = len(b) - 1
    if len(delay) != mglsadf_delay_length(order, stage):
        raise ValueError("inconsistent delay length")

    return _mglsadf(x, &b[0], order, alpha, stage, &delay[0])


### Utils ###

def phidf(x, order, alpha,
          np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
    if len(delay) != order + 1:
        raise ValueError("inconsistent order or delay")

    _phidf(x, order, alpha, &delay[0])


def lspcheck(np.ndarray[np.float64_t, ndim=1, mode="c"] lsp not None):
    cdef int ret = _lspcheck(&lsp[0], len(lsp) - 1)
    return ret
