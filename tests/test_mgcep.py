# coding: utf-8

import numpy as np
import pysptk
from warnings import warn
from nose.tools import raises


def windowed_dummy_data(N):
    np.random.seed(98765)
    return pysptk.hanning(N) * np.random.randn(N)


def test_mcep():
    x = windowed_dummy_data(1024)

    def __test(order, alpha):
        mc = pysptk.mcep(x, order, alpha)
        assert np.all(np.isfinite(mc))

    for order in [15, 20, 25]:
        for alpha in [0.0, 0.35, 0.41]:
            yield __test, order, alpha


def test_mcep_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_itype(itype=0):
        pysptk.mcep(x, itype=itype)

    yield raises(ValueError)(__test_itype), -1
    yield raises(ValueError)(__test_itype), 5

    def __test_eps(etype=0, eps=0.0):
        pysptk.mcep(x, etype=etype, eps=eps)

    yield raises(ValueError)(__test_eps), 0, -1.0
    yield raises(ValueError)(__test_eps), -1
    yield raises(ValueError)(__test_eps), -3
    yield raises(ValueError)(__test_eps), 1, -1.0
    yield raises(ValueError)(__test_eps), 2, -1.0

    def __test_min_det(min_det):
        pysptk.mcep(x, min_det=min_det)

    yield raises(ValueError)(__test_min_det), -1.0


@raises(RuntimeError)
def test_mcep_failure():
    pysptk.mcep(np.ones(256), 40, 0.41)


def test_gcep():
    x = windowed_dummy_data(1024)

    def __test(order, gamma):
        gc = pysptk.gcep(x, order, gamma)
        assert np.all(np.isfinite(gc))

    for order in [15, 20, 25]:
        for gamma in [0.0, -0.25, -0.5]:
            yield __test, order, gamma


def test_gcep_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_gamma(gamma):
        pysptk.gcep(x, gamma=gamma)

    yield raises(ValueError)(__test_gamma), 0.1
    yield raises(ValueError)(__test_gamma), -2.1

    def __test_itype(itype=0):
        pysptk.gcep(x, itype=itype)

    yield raises(ValueError)(__test_itype), -1
    yield raises(ValueError)(__test_itype), 5

    def __test_eps(etype=0, eps=0.0):
        pysptk.gcep(x, etype=etype, eps=eps)

    yield raises(ValueError)(__test_eps), 0, -1.0
    yield raises(ValueError)(__test_eps), -1
    yield raises(ValueError)(__test_eps), -3
    yield raises(ValueError)(__test_eps), 1, -1.0
    yield raises(ValueError)(__test_eps), 2, -1.0

    def __test_min_det(min_det):
        pysptk.gcep(x, min_det=min_det)

    yield raises(ValueError)(__test_min_det), -1.0


@raises(RuntimeError)
def test_gcep_failure():
    pysptk.gcep(np.ones(256), 40, 0.0)


def test_mgcep():
    x = windowed_dummy_data(1024)

    def __test(order, alpha, gamma):
        mgc = pysptk.mgcep(x, order, alpha, gamma)
        assert np.all(np.isfinite(mgc))

    for order in [15, 20, 25]:
        for alpha in [0.0, 0.35, 0.41]:
            for gamma in [0.0, -0.25, -0.5]:
                yield __test, order, alpha, gamma


def test_mgcep_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_gamma(gamma):
        pysptk.mgcep(x, gamma=gamma)

    yield raises(ValueError)(__test_gamma), 0.1
    yield raises(ValueError)(__test_gamma), -2.1

    def __test_itype(itype=0):
        pysptk.mgcep(x, itype=itype)

    yield raises(ValueError)(__test_itype), -1
    yield raises(ValueError)(__test_itype), 5

    def __test_eps(etype=0, eps=0.0):
        pysptk.mgcep(x, etype=etype, eps=eps)

    yield raises(ValueError)(__test_eps), 0, -1.0
    yield raises(ValueError)(__test_eps), -1
    yield raises(ValueError)(__test_eps), -3
    yield raises(ValueError)(__test_eps), 1, -1.0
    yield raises(ValueError)(__test_eps), 2, -1.0

    def __test_min_det(min_det):
        pysptk.mgcep(x, min_det=min_det)

    yield raises(ValueError)(__test_min_det), -1.0

    def __test_otype(otype=0):
        pysptk.mgcep(x, otype=otype)

    yield raises(ValueError)(__test_otype), -1
    yield raises(ValueError)(__test_otype), 6


@raises(RuntimeError)
def test_mgcep_failure():
    pysptk.mgcep(np.ones(256))


def test_uels():
    x = windowed_dummy_data(1024)

    def __test(order):
        c = pysptk.uels(x, order)
        assert np.all(np.isfinite(c))

    for order in [15, 20, 25]:
        yield __test, order


def test_uels_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_itype(itype=0):
        pysptk.uels(x, itype=itype)

    yield raises(ValueError)(__test_itype), -1
    yield raises(ValueError)(__test_itype), 5

    def __test_eps(etype=0, eps=0.0):
        pysptk.uels(x, etype=etype, eps=eps)

    yield raises(ValueError)(__test_eps), 0, -1.0
    yield raises(ValueError)(__test_eps), -1
    yield raises(ValueError)(__test_eps), -3
    yield raises(ValueError)(__test_eps), 1, -1.0
    yield raises(ValueError)(__test_eps), 2, -1.0


@raises(RuntimeError)
def test_uels_failure():
    pysptk.uels(np.ones(256), 40)


def test_fftcep():
    x = windowed_dummy_data(1024)
    logsp = np.log(np.abs(np.fft.rfft(x)) + 1.0e-6)

    def __test(order):
        c = pysptk.fftcep(logsp, order)
        assert np.all(np.isfinite(c))

    for order in [15, 20, 25]:
        yield __test, order


def test_lpc():
    x = windowed_dummy_data(1024)

    def __test(order):
        a = pysptk.lpc(x, order)
        assert np.all(np.isfinite(a))

    for order in [15, 20, 25]:
        yield __test, order


def test_lpc_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_min_det(min_det):
        pysptk.lpc(x, min_det=min_det)

    yield raises(ValueError)(__test_min_det), -1.0


@raises(RuntimeError)
def test_lpc_failure():
    pysptk.lpc(np.zeros(256), 40)
