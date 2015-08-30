# coding: utf-8

import numpy as np
import six
import pysptk
from warnings import warn
from nose.tools import raises


def windowed_dummy_data(N):
    np.random.seed(98765)
    return pysptk.hanning(N) * np.random.randn(N)


def test_agexp():
    assert pysptk.agexp(1, 1, 1) == 5.0
    assert pysptk.agexp(1, 2, 3) == 18.0


def test_gexp():
    assert pysptk.gexp(1, 1) == 2.0
    assert pysptk.gexp(2, 4) == 3.0


def test_glog():
    assert pysptk.glog(1, 2) == 1.0
    assert pysptk.glog(2, 3) == 4.0


def test_mseq():
    for i in six.moves.range(0, 100):
        assert np.isfinite(pysptk.mseq())


def test_acep():
    def __test(order, pd):
        x = windowed_dummy_data(64)
        c = np.zeros(order + 1)
        for v in x:
            pysptk.acep(v, c, pd=pd)
            assert np.all(np.isfinite(c))

    for order in [20, 22, 25]:
        for pd in [4, 5]:
            yield __test, order, pd


def test_agcep():
    def __test(order, stage):
        x = windowed_dummy_data(64)
        c = np.zeros(order + 1)
        for v in x:
            pysptk.agcep(v, c, stage=stage)
            assert np.all(np.isfinite(c))

    for order in [20, 22, 25]:
        for stage in six.moves.range(1, 10):
            yield __test, order, stage


def test_amcep():
    def __test(order, stage):
        x = windowed_dummy_data(64)
        c = np.zeros(order + 1)
        for v in x:
            pysptk.amcep(v, c, alpha=alpha)
            assert np.all(np.isfinite(c))

    for order in [20, 22, 25]:
        for alpha in [0.0, 0.35, 0.5]:
            yield __test, order, alpha


def test_swipe():

    def __test(x, fs, hopsize, otype):
        f0 = pysptk.swipe(x, fs, hopsize, otype=otype)
        assert np.all(np.isfinite(f0))

    np.random.seed(98765)
    fs = 16000
    hopsize = 80
    x = np.random.rand(16000)

    for otype in six.moves.range(0, 3):
        yield __test, x, fs, hopsize, otype

    yield raises(ValueError)(__test), x, fs, hopsize, -1
    yield raises(ValueError)(__test), x, fs, hopsize, 3


def test_windows():
    def __test(f, N, normalize):
        w = f(N, normalize)
        assert np.all(np.isfinite(w))

    from pysptk import blackman, hanning, hamming
    from pysptk import bartlett, trapezoid, rectangular

    for f in [blackman, hanning, hamming, bartlett, trapezoid, rectangular]:
        for n in [16, 128, 256, 1024, 2048, 4096]:
            yield __test, f, n, 1

    for f in [blackman, hanning, hamming, bartlett, trapezoid, rectangular]:
        yield raises(ValueError)(__test), f, 256, -1
        yield raises(ValueError)(__test), f, 256, 3


def test_mcep():
    x = windowed_dummy_data(1024)

    def __test(order, alpha):
        mc = pysptk.mcep(x, order, alpha)
        assert np.all(np.isfinite(mc))

    for order in [15, 20, 25]:
        for alpha in [0.0, 0.35, 0.41]:
            yield __test, order, alpha


def test_gcep():
    x = windowed_dummy_data(1024)

    def __test(order, gamma):
        gc = pysptk.gcep(x, order, gamma)
        assert np.all(np.isfinite(gc))

    for order in [15, 20, 25]:
        for gamma in [0.0, -0.25, -0.5]:
            yield __test, order, gamma


def test_mgcep():
    x = windowed_dummy_data(1024)

    def __test(order, alpha, gamma):
        mgc = pysptk.mgcep(x, order, alpha, gamma)
        assert np.all(np.isfinite(mgc))

    for order in [15, 20, 25]:
        for alpha in [0.0, 0.35, 0.41]:
            for gamma in [0.0, -0.25, -0.5]:
                yield __test, order, alpha, gamma


def test_uels():
    x = windowed_dummy_data(1024)

    def __test(order):
        c = pysptk.uels(x, order)
        assert np.all(np.isfinite(c))

    for order in [15, 20, 25]:
        yield __test, order


def test_fftcep():
    x = windowed_dummy_data(1024)
    logsp = np.log(np.abs(np.fft.rfft(x)) + 1.0e-6)

    def __test(order):
        c = pysptk.fftcep(logsp, order)
        assert np.all(np.isfinite(c))

    warn("TODO: fix memory corruption in fftcep")
    # for order in [15, 20, 25]:
    #    yield __test, order


def test_lpc():
    x = windowed_dummy_data(1024)

    def __test(order):
        a = pysptk.lpc(x, order)
        assert np.all(np.isfinite(a))

    for order in [15, 20, 25]:
        yield __test, order
