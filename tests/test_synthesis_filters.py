# coding: utf-8

import numpy as np
import six
import pysptk
from nose.tools import raises


def __test_filt_base(f, order, delay, *args):
    np.random.seed(98765)
    dummy_input = np.random.rand(1024)
    dummy_mgc = np.random.rand(order + 1)

    for x in dummy_input:
        assert np.isfinite(f(x, dummy_mgc, *args, delay=delay))
        assert np.all(np.isfinite(delay))


def test_poledf():
    for order in [20, 25, 30]:
        delay = pysptk.poledf_delay(order)
        yield __test_filt_base, pysptk.poledf, order, delay


@raises(ValueError)
def test_poledf_invalid_delay_length():
    pysptk.poledf(0.0, np.ones(10), np.ones(1))


def test_lmadf():
    for order in [20, 25, 30]:
        for pd in [4, 5]:
            delay = pysptk.lmadf_delay(order, pd)
            yield __test_filt_base, pysptk.lmadf, order, delay, pd


@raises(ValueError)
def test_lmadf_invalid_delay_length():
    pysptk.lmadf(0.0, np.ones(10), 5, np.ones(1))


@raises(ValueError)
def test_lmadf_invalid_pade():
    pysptk.lmadf(0.0, np.ones(10), 3, np.ones(1))


def test_lspdf():
    for order in [20, 25, 30]:
        delay = pysptk.lspdf_delay(order)
        yield __test_filt_base, pysptk.lspdf, order, delay


def test_lspdf_invalid_delay_length():
    def __test(length):
        pysptk.lspdf(0.0, np.ones(length), np.ones(1))

    # even
    yield raises(ValueError)(__test), 10
    # odd
    yield raises(ValueError)(__test), 9


def test_ltcdf():
    for order in [20, 25, 30]:
        delay = pysptk.ltcdf_delay(order)
        yield __test_filt_base, pysptk.ltcdf, order, delay


@raises(ValueError)
def test_ltcdf_invalid_delay_length():
    pysptk.ltcdf(0.0, np.ones(10), np.ones(1))


def test_glsadf():
    for order in [20, 25, 30]:
        for stage in six.moves.range(1, 7):
            delay = pysptk.glsadf_delay(order, stage)
            yield __test_filt_base, pysptk.glsadf, order, delay, stage


@raises(ValueError)
def test_glsadf_invalid_delay_length():
    pysptk.glsadf(0.0, np.ones(10), 1, np.ones(1))


@raises(ValueError)
def test_glsadf_invalid_stage():
    pysptk.glsadf(0.0, np.ones(10), 0, np.ones(1))


def test_mlsadf():
    for order in [20, 25, 30]:
        for alpha in [0.0, 0.35, 0.5]:
            for pd in [4, 5]:
                delay = pysptk.mlsadf_delay(order, pd)
                yield __test_filt_base, pysptk.mlsadf, order, delay, alpha, pd


@raises(ValueError)
def test_mlsadf_invalid_delay_length():
    pysptk.mlsadf(0.0, np.ones(10), 0.41, 5, np.ones(1))


@raises(ValueError)
def test_mlsadf_invalid_pade():
    pysptk.mlsadf(0.0, np.ones(10), 0.41, 3, np.ones(1))


def test_mglsadf():
    for order in [20, 25, 30]:
        for alpha in [0.0, 0.35, 0.5]:
            for stage in six.moves.range(1, 7):
                delay = pysptk.mglsadf_delay(order, stage)
                yield __test_filt_base, pysptk.mglsadf, order, delay, alpha, stage


@raises(ValueError)
def test_mglsadf_invalid_delay_length():
    pysptk.mglsadf(0.0, np.ones(10), 0.41, 15, np.ones(1))


@raises(ValueError)
def test_mglsadf_invalid_stage():
    pysptk.mglsadf(0.0, np.ones(10), 0.41, 0, np.ones(1))
