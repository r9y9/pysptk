# coding: utf-8

import numpy as np
import six
import pysptk
from nose.tools import raises


def windowed_dummy_data(N):
    np.random.seed(98765)
    return pysptk.hanning(N) * np.random.randn(N)


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

    # invalid pade
    yield raises(ValueError)(__test), 20, 3
    yield raises(ValueError)(__test), 20, 6


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

    # invalid stage
    yield raises(ValueError)(__test), 20, 0


def test_amcep():
    def __test(order, stage, pd=5):
        x = windowed_dummy_data(64)
        c = np.zeros(order + 1)
        for v in x:
            pysptk.amcep(v, c, alpha=alpha, pd=pd)
            assert np.all(np.isfinite(c))

    for order in [20, 22, 25]:
        for alpha in [0.0, 0.35, 0.5]:
            yield __test, order, alpha

    # invalid pade
    yield raises(ValueError)(__test), 20, 0.35, 3
    yield raises(ValueError)(__test), 20, 0.35, 6
