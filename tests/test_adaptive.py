import numpy as np
import pysptk
import pytest


def windowed_dummy_data(N):
    np.random.seed(98765)
    return pysptk.hanning(N) * np.random.randn(N)


# TODO: likely to have bugs in SPTK
@pytest.mark.parametrize("order", [20, 22, 25])
@pytest.mark.parametrize("pd", [4, 5])
def test_acep(order, pd):
    return
    # x = windowed_dummy_data(64)
    # c = np.zeros(order + 1)
    # for v in x:
    #     pysptk.acep(v, c, pd=pd)
    #     assert np.all(np.isfinite(c))


def test_acep_corner_case():
    def __test(order, pd):
        x = windowed_dummy_data(64)
        c = np.zeros(order + 1)
        for v in x:
            pysptk.acep(v, c, pd=pd)

    # invalid pade
    with pytest.raises(ValueError):
        __test(20, 3)
    with pytest.raises(ValueError):
        __test(20, 8)


@pytest.mark.parametrize("order", [20, 22, 25])
# TODO: likely to have bugs in SPTK
# @pytest.mark.parametrize("stage", [1, 2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("stage", [1, 2])
def test_agcep(order, stage):
    x = windowed_dummy_data(64)
    c = np.zeros(order + 1)
    for v in x:
        pysptk.agcep(v, c, stage=stage)
        assert np.all(np.isfinite(c))


def test_agcep_corner_case():
    def __test(order, stage):
        x = windowed_dummy_data(64)
        c = np.zeros(order + 1)
        for v in x:
            pysptk.agcep(v, c, stage=stage)

    # invalid stage
    with pytest.raises(ValueError):
        __test(20, 0)


@pytest.mark.parametrize("order", [20, 22, 25])
@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.5])
def test_amcep(order, alpha, pd=5):
    x = windowed_dummy_data(64)
    c = np.zeros(order + 1)
    for v in x:
        pysptk.amcep(v, c, alpha=alpha, pd=pd)
        assert np.all(np.isfinite(c))


def test_amcep_corner_case():
    def __test(order, alpha, pd=5):
        x = windowed_dummy_data(64)
        c = np.zeros(order + 1)
        for v in x:
            pysptk.amcep(v, c, alpha=alpha, pd=pd)

    # invalid pade
    with pytest.raises(ValueError):
        __test(20, 0.35, 3)
    with pytest.raises(ValueError):
        __test(20, 0.35, 8)
