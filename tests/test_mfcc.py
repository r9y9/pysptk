# coding: utf-8

import numpy as np
import pysptk
from nose.tools import raises


def test_mfcc_options():
    np.random.seed(98765)
    dummy_input = np.random.rand(512)

    # with c0
    cc = pysptk.mfcc(dummy_input, 12, czero=True)
    assert len(cc) == 13

    # wth c0 + power
    cc = pysptk.mfcc(dummy_input, 12, czero=True, power=True)
    assert len(cc) == 14

    # with power
    cc = pysptk.mfcc(dummy_input, 12, power=True)
    assert len(cc) == 13


def test_mfcc_num_filterbanks():
    def __test(n):
        np.random.seed(98765)
        dummy_input = np.random.rand(512)
        cc = pysptk.mfcc(dummy_input, 20, num_filterbanks=n)
        assert np.all(np.isfinite(cc))

    for n in [21, 23, 25]:
        yield __test, n

    for n in [19, 20]:
        yield raises(ValueError)(__test), n


def test_mfcc():
    def __test(length, order):
        np.random.seed(98765)
        dummy_input = np.random.rand(length)
        cc = pysptk.mfcc(dummy_input, order, czero=True, power=True)
        assert np.all(np.isfinite(cc))

    for length in [256, 512, 1024, 2048, 4096]:
        for order in [12, 14, 16, 18]:
            yield __test, length, order
