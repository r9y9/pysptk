import numpy as np
import pysptk
import pytest


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


@pytest.mark.parametrize("n", [21, 23, 25])
def test_mfcc_num_filterbanks(n):
    def __test(n):
        np.random.seed(98765)
        dummy_input = np.random.rand(512)
        cc = pysptk.mfcc(dummy_input, 20, num_filterbanks=n)
        assert np.all(np.isfinite(cc))

    __test(n)


def test_mgcc_num_filterbanks_corner_case():
    def __test(n):
        np.random.seed(98765)
        dummy_input = np.random.rand(512)
        pysptk.mfcc(dummy_input, 20, num_filterbanks=n)

    for n in [19, 20]:
        with pytest.raises(ValueError):
            __test(n)


@pytest.mark.parametrize("order", [12, 14, 16, 18])
@pytest.mark.parametrize("length", [256, 512, 1024, 2048, 4096])
def test_mfcc(order, length):
    def __test(length, order):
        np.random.seed(98765)
        dummy_input = np.random.rand(length)
        cc = pysptk.mfcc(dummy_input, order, czero=True, power=True)
        assert np.all(np.isfinite(cc))

    __test(length, order)
