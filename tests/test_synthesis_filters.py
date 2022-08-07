import numpy as np
import pysptk
import pytest


def __test_filt_base(f, order, delay, *args):
    np.random.seed(98765)
    dummy_input = np.random.rand(1024)
    dummy_mgc = np.random.rand(order + 1)

    for x in dummy_input:
        assert np.isfinite(f(x, dummy_mgc, *args, delay=delay))
        assert np.all(np.isfinite(delay))


@pytest.mark.parametrize("order", [20, 25, 30])
def test_poledf(order):
    delay = pysptk.poledf_delay(order)
    __test_filt_base(pysptk.poledf, order, delay)
    __test_filt_base(pysptk.poledft, order, delay)


def test_poledf_invalid_delay_length():
    with pytest.raises(ValueError):
        pysptk.poledf(0.0, np.ones(10), np.ones(1))


@pytest.mark.parametrize("order", [20, 25, 30])
@pytest.mark.parametrize("pd", [4, 5])
def test_lmadf(order, pd):
    delay = pysptk.lmadf_delay(order, pd)
    __test_filt_base(pysptk.lmadf, order, delay, pd)


def test_lmadf_invalid_delay_length():
    with pytest.raises(ValueError):
        pysptk.lmadf(0.0, np.ones(10), 5, np.ones(1))


def test_lmadf_invalid_pade():
    with pytest.raises(ValueError):
        pysptk.lmadf(0.0, np.ones(10), 3, np.ones(1))


@pytest.mark.parametrize("order", [20, 25, 30])
def test_lspdf(order):
    delay = pysptk.lspdf_delay(order)
    __test_filt_base(pysptk.lspdf, order, delay)


def test_lspdf_invalid_delay_length():
    def __test(length):
        pysptk.lspdf(0.0, np.ones(length), np.ones(1))

    # even
    with pytest.raises(ValueError):
        __test(10)
    # odd
    with pytest.raises(ValueError):
        __test(9)


@pytest.mark.parametrize("order", [20, 25, 30])
def test_ltcdf(order):
    delay = pysptk.ltcdf_delay(order)
    __test_filt_base(pysptk.ltcdf, order, delay)


def test_ltcdf_invalid_delay_length():
    with pytest.raises(ValueError):
        pysptk.ltcdf(0.0, np.ones(10), np.ones(1))


@pytest.mark.parametrize("order", [20, 22, 25])
@pytest.mark.parametrize("stage", [1, 2, 3, 4, 5, 6])
def test_glsadf(order, stage):
    delay = pysptk.glsadf_delay(order, stage)
    __test_filt_base(pysptk.glsadf, order, delay, stage)
    __test_filt_base(pysptk.glsadft, order, delay, stage)


def test_glsadf_invalid_delay_length():
    with pytest.raises(ValueError):
        pysptk.glsadf(0.0, np.ones(10), 1, np.ones(1))


def test_glsadf_invalid_stage():
    with pytest.raises(ValueError):
        pysptk.glsadf(0.0, np.ones(10), 0, np.ones(1))


@pytest.mark.parametrize("order", [20, 25, 30])
@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.5])
@pytest.mark.parametrize("pd", [4, 5])
def test_mlsadf(order, alpha, pd):
    delay = pysptk.mlsadf_delay(order, pd)
    __test_filt_base(pysptk.mlsadf, order, delay, alpha, pd)
    __test_filt_base(pysptk.mlsadft, order, delay, alpha, pd)


def test_mlsadf_invalid_delay_length():
    with pytest.raises(ValueError):
        pysptk.mlsadf(0.0, np.ones(10), 0.41, 5, np.ones(1))


def test_mlsadf_invalid_pade():
    with pytest.raises(ValueError):
        pysptk.mlsadf(0.0, np.ones(10), 0.41, 3, np.ones(1))


@pytest.mark.parametrize("order", [20, 25, 30])
@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.5])
@pytest.mark.parametrize("stage", [1, 2, 3, 4, 5, 6])
def test_mglsadf(order, alpha, stage):
    delay = pysptk.mglsadf_delay(order, stage)
    __test_filt_base(pysptk.mglsadf, order, delay, alpha, stage)
    __test_filt_base(pysptk.mglsadft, order, delay, alpha, stage)


def test_mglsadf_invalid_delay_length():
    with pytest.raises(ValueError):
        pysptk.mglsadf(0.0, np.ones(10), 0.41, 15, np.ones(1))


def test_mglsadf_invalid_stage():
    with pytest.raises(ValueError):
        pysptk.mglsadf(0.0, np.ones(10), 0.41, 0, np.ones(1))
