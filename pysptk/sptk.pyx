# coding: utf-8
# cython: boundscheck=True, wraparound=True

import numpy as np
cimport numpy as np

cimport cython
cimport sptk

from warnings import warn
from pysptk import assert_gamma, assert_fftlen, assert_pade


### Library routines ###

def agexp(r, x, y):
    return _agexp(r, x, y)


def gexp(r, x):
    """Generalized exponential function

    Parameters
    ----------
    r : float
        gamma

    x : float
        arg

    Returns
    -------
    value

    """
    return _gexp(r, x)


def glog(r, x):
    """Generalized logarithmic function

    Parameters
    ----------
    r : float
        gamma
    x : float
        arg

    Returns
    -------
    value

    """
    return _glog(r, x)


def mseq():
    return _mseq()


### Adaptive mel-generalized cepstrum analysis ###

def acep(x, np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
         lambda_coef=0.98, step=0.1, tau=0.9, pd=4, eps=1.0e-6):
    assert_pade(pd)
    cdef int order = len(c) - 1
    cdef double prederr
    prederr = _acep(x, &c[0], order, lambda_coef, step, tau, pd, eps)
    return prederr


def agcep(x, np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
          stage=1,
          lambda_coef=0.98, step=0.1, tau=0.9, eps=1.0e-6):
    if stage < 1:
        raise ValueError("stage >= 1 (-1 <= gamma < 0)")
    cdef int order = len(c) - 1
    cdef double prederr
    prederr = _agcep(x, &c[0], order, stage, lambda_coef, step, tau, eps)
    return prederr


def amcep(x, np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
          alpha=0.41,
          lambda_coef=0.98, step=0.1, tau=0.0, pd=4, eps=1.0e-6):
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
        order of mel-cepstrum

    alpha : float
        all pass constant

    miniter : int
        minimum number of iteration

    maxiter : int
        maximum number of iteration

    threshold : float
        threshold in theq

    etype : int
        type of parameter `eps`
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

    eps : float
        initial value for log-periodogram or floor of periodogram in db

    min_det : float
        mimimum value of the determinant of normal matrix

    itype : float
        input data type:
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

    Returns
    -------
    mc : array, shape (`order + 1`)

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
        order of mel-generalized cepstrum

    alpha : float
        all pass constant

    num_recursions : int (default=len(windowed)-1)
        number of recursions

    miniter : int
        minimum number of iteration

    maxiter : int
        maximum number of iteration

    threshold : float
        threshold

    etype : int
        type of paramter `e`
             (0) not used
             (1) initial value of log-periodogram
             (2) floor of periodogram in db

    eps : float
        initial value for log-periodogram or floor of periodogram in db

    min_det : float
        mimimum value of the determinant of normal matrix

    itype : float
        input data type:
            (0) windowed signal
            (1) log amplitude in db
            (2) log amplitude
            (3) amplitude
            (4) periodogram

    otype : int
        output data type
            (0) mel generalized cepstrum: (c~0...c~m)
            (1) MGLSA filter coefficients: b0...bm
            (2) K~,c~'1...c~'m
            (3) K,b'1...b'm
            (4) K~,g*c~'1...g*c~'m
            (5) K,g*b'1...g*b'm

    Returns
    -------
    mgc : array, shape (`order + 1`)

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


def lpc2c(np.ndarray[np.float64_t, ndim=1, mode="c"] src_lpc not None,
          dst_order):
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_ceps
    cdef int src_order = len(src_lpc) - 1
    dst_ceps = np.zeros(dst_order + 1, dtype=np.float64)
    _lpc2c(&src_lpc[0], src_order, &dst_ceps[0], dst_order)
    return dst_ceps


@cython.boundscheck(False)
@cython.wraparound(False)
def lpc2lsp(np.ndarray[np.float64_t, ndim=1, mode="c"] src_lpc not None,
            numsp=512, maxiter=4, eps=1.0e-6, loggain=False, otype=0,
            fs=None):
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_lsp
    cdef int order = len(src_lpc) - 1
    dst_lsp = np.zeros_like(src_lpc)
    _lpc2lsp(&src_lpc[0], &dst_lsp[0], order, numsp, maxiter, eps)

    if otype == 0:
        dst_lsp[1:] *= 2 * np.pi
    elif otype == 2 or otype == 3:
        if fs is None:
            raise ValueError("fs must be specified when otype == 2 or 3")
        dst_lsp[1:] *= fs

    if otype == 3:
        dst_lsp[1:] *= 1000.0

    if loggain:
        dst_lsp[0] = np.log(src_lpc[0])
    else:
        dst_lsp[0] = src_lpc[0]

    return dst_lsp


def lpc2par(np.ndarray[np.float64_t, ndim=1, mode="c"] src_lpc not None):
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_par
    dst_par = np.zeros_like(src_lpc)
    cdef int order = len(src_lpc) - 1
    _lpc2par(&src_lpc[0], &dst_par[0], order)
    return dst_par


def par2lpc(np.ndarray[np.float64_t, ndim=1, mode="c"] src_par not None):
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] dst_lpc
    dst_lpc = np.zeros_like(src_par)
    cdef int order = len(src_par) - 1
    _par2lpc(&src_par[0], &dst_lpc[0], order)
    return dst_lpc


# assume lsp has loggain at src_lsp[0]
def lsp2sp(np.ndarray[np.float64_t, ndim=1, mode="c"] src_lsp not None,
           fftlen=256):
    assert_fftlen(fftlen)
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] sp
    cdef int sp_length = (fftlen >> 1) + 1
    sp = np.zeros(sp_length, dtype=np.float64)
    cdef int order = len(src_lsp) - 1
    _lsp2sp(&src_lsp[0], order, &sp[0], sp_length, 1)
    return sp


### Mel-generalized cepstrum conversions ###

def mc2b(np.ndarray[np.float64_t, ndim=1, mode="c"] mc not None,
         alpha=0.35):
    """Mel-cepsrum to MLSA filter coefficients

    Parameters
    ----------
    mc : array, shape (`order of mel-cepstrum` + 1)
         mel-cepstrum

    alpha : float
         all-pass constant

    Returns
    -------
    MLSA filter coefficients : array, shape(=mc)

    """
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] b
    b = np.zeros_like(mc)
    cdef int order = len(mc) - 1
    _mc2b(&mc[0], &b[0], order, alpha)
    return b


def b2mc(np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
         alpha=0.35):
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
          dst_order=None,
          fftlen=256):
    assert_fftlen(fftlen)
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] r
    cdef int src_order = len(c) - 1
    if dst_order is None:
        dst_order = src_order
    r = np.zeros(dst_order + 1, dtype=np.float64)
    _c2acr(&c[0], src_order, &r[0], dst_order, fftlen)
    return r


def c2ir(np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
         length=256):
    """Cepstrum to impulse response

    Parameters
    ----------
    c : array, shape (`order + 1`)
         cepstrum
    length : int
         length of impulse response

    Returns
    -------
    impulse response, shape (`length`)

    """

    cdef int order = len(c)  # NOT len(c) - 1
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] h
    h = np.zeros(length, dtype=np.float64)
    _c2ir(&c[0], order, &h[0], length)
    return h


def ic2ir(np.ndarray[np.float64_t, ndim=1, mode="c"] h not None,
          order=25):
    cdef int length = len(h)
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] c
    c = np.zeros(order + 1, dtype=np.float64)
    _ic2ir(&h[0], length, &c[0], len(c))
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
def c2ndps(np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
           fftlen=256):
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
    # assuming the lenght of ndps is fftlen/2+1
    cdef int fftlen = (len(ndps) - 1) << 1
    cdef np.ndarray[np.float64_t, ndim = 1, mode = "c"] c
    assert_fftlen(fftlen)
    c = np.zeros(order + 1, dtype=np.float64)
    _ndps2c(&ndps[0], fftlen, &c[0], order)
    return ndps


def gc2gc(np.ndarray[np.float64_t, ndim=1, mode="c"] src_ceps not None,
          src_gamma=0.0, dst_order=None, dst_gamma=0.0):
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
    c : array, shape (`order of cepstrum` + 1)
        generalized cepstrum
    gamma : float
        gamma

    Returns
    -------
    normalized generalized cepstrum : array, shape(=ceps)

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
    c : array, shape (`order of cepstrum` + 1)
        normalized generalized cepstrum
    gamma : float
        gamma

    Returns
    -------
    generalized cepstrum : array, shape(=ceps)

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
    ceps : array, shape (`order of cepstrum ` + 1)
        cepstrum
    order : int
        desired order of cepstrum that will be converted
    alpha : float
        all-pass constant

    Returns
    -------
    dst_ceps : array, shape(`order` + 1)

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
    assert_gamma(gamma)

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
    """SWIPE - A Saw-tooth Waveform Inspired Pitch Estimation

    Parameters
    ----------
    x : array, shape (`the number of audio samples`)
        A whole signal

    fs : int
        samplerate

    hopsize : int
        hop size

    min : float (default=50.0)
        minimum fundamental frequency

    max : float (default=800.0)
        maximum fundamental frequency

    threshold : float (default=0.3)
        voice/unvoiced threshold

    otype : int (default=1)
        output format (0) pitch (1) f0 (2) log(f0)

    Returns
    -------
    f0  : array, shape(`len(x)/frame_shift+1`)

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
         window length
    normalize : int
         0 : don't normalize
         1 : normalize by power
         2 : normalize by magnitude

    Returns
    -------
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
         window length
    normalize : int
         0 : don't normalize
         1 : normalize by power
         2 : normalize by magnitude

    Returns
    -------
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
         window length
    normalize : int
         0 : don't normalize
         1 : normalize by power
         2 : normalize by magnitude

    Returns
    -------
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
         window length
    normalize : int
         0 : don't normalize
         1 : normalize by power
         2 : normalize by magnitude

    Returns
    -------
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
         window length
    normalize : int
         0 : don't normalize
         1 : normalize by power
         2 : normalize by magnitude

    Returns
    -------
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
         window length
    normalize : int
         0 : don't normalize
         1 : normalize by power
         2 : normalize by magnitude

    Returns
    -------
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
    return np.zeros(poledf_delay_length(order))


def poledf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] a not None,
           np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
    cdef int order = len(a) - 1
    if len(delay) != poledf_delay_length(order):
        raise ValueError("inconsistent delay length")

    return _poledf(x, &a[0], order, &delay[0])


def lmadf_delay_length(order, pd):
    return 2 * pd * (order + 1)


def lmadf_delay(order, pd):
    return np.zeros(lmadf_delay_length(order, pd))


def lmadf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
          pd,
          np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
    assert_pade(pd)

    cdef int order = len(b) - 1
    if len(delay) != lmadf_delay_length(order, pd):
        raise ValueError("inconsistent delay length")

    return _lmadf(x, &b[0], order, pd, &delay[0])


def lspdf_delay_length(order):
    return 2 * order + 1


def lspdf_delay(order):
    return np.zeros(lspdf_delay_length(order))


def lspdf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] f not None,
          np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
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
    return np.zeros(ltcdf_delay_length(order))


def ltcdf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] k not None,
          np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
    cdef int order = len(k) - 1
    if len(delay) != ltcdf_delay_length(order):
        raise ValueError("inconsistent delay length")

    return _ltcdf(x, &k[0], order, &delay[0])


def glsadf_delay_length(order, stage):
    return order * (stage + 1) + 1


def glsadf_delay(order, stage):
    return np.zeros(glsadf_delay_length(order, stage))


def glsadf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] c not None,
           stage,
           np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
    if stage < 1:
        raise ValueError("stage >= 1 (-1 <= gamma < 0)")

    cdef int order = len(c) - 1
    if len(delay) != glsadf_delay_length(order, stage):
        raise ValueError("inconsistent delay length")

    return _glsadf(x, &c[0], order, stage, &delay[0])


def mlsadf_delay_length(order, pd):
    return 3 * (pd + 1) + pd * (order + 2)


def mlsadf_delay(order, pd):
    return np.zeros(mlsadf_delay_length(order, pd))


def mlsadf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
           alpha, pd,
           np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
    assert_pade(pd)

    cdef int order = len(b) - 1
    if len(delay) != mlsadf_delay_length(order, pd):
        raise ValueError("inconsistent delay length")

    return _mlsadf(x, &b[0], order, alpha, pd, &delay[0])


def mglsadf_delay_length(order, stage):
    return (order + 1) * stage


def mglsadf_delay(order, stage):
    return np.zeros(mglsadf_delay_length(order, stage))


def mglsadf(x, np.ndarray[np.float64_t, ndim=1, mode="c"] b not None,
            alpha, stage,
            np.ndarray[np.float64_t, ndim=1, mode="c"] delay not None):
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
