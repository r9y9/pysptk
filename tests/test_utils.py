import numpy as np
import pysptk
import pytest
from pysptk.util import apply_along_last_axis, automatic_type_conversion, mcepalpha


def test_assert_gamma():
    def __test(gamma):
        pysptk.util.assert_gamma(gamma)

    for gamma in [-2.0, 0.1]:
        with pytest.raises(ValueError):
            __test(gamma)


def test_assert_pade():
    def __test(pade):
        pysptk.util.assert_pade(pade)

    for pade in [3, 8]:
        with pytest.raises(ValueError):
            __test(pade)


def test_assert_fftlen():
    def __test(fftlen):
        pysptk.util.assert_fftlen(fftlen)

    for fftlen in [255, 257]:
        with pytest.raises(ValueError):
            __test(fftlen)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("alpha", [0.35, 0.41, 0.5])
def test_phidf(order, alpha):
    np.random.seed(98765)
    dummy_input = np.random.rand(64)
    delay = np.zeros(order + 1)
    for x in dummy_input:
        pysptk.phidf(x, order, alpha, delay)
        assert np.all(np.isfinite(delay))


@pytest.mark.parametrize("order", [15, 20, 25, 30])
def test_lspcheck(order):
    np.random.seed(98765)
    lsp = np.random.rand(order + 1)
    pysptk.lspcheck(lsp)
    # TODO: valid check


def test_example_audio_file():
    from os.path import exists

    path = pysptk.util.example_audio_file()
    assert exists(path)


def test_mcepalpha():
    assert np.isclose(mcepalpha(8000), 0.312)
    assert np.isclose(mcepalpha(11025), 0.357)
    assert np.isclose(mcepalpha(16000), 0.41)
    assert np.isclose(mcepalpha(22050), 0.455)
    assert np.isclose(mcepalpha(44100), 0.544)
    assert np.isclose(mcepalpha(48000), 0.554)


def test_automatic_type_conversion():
    @automatic_type_conversion
    def f(x):
        return x

    for dtype in [np.float32, np.float16, np.float64]:
        x = np.ones(10, dtype=dtype)
        y = f(x)
        assert y.dtype == x.dtype
        y = f(x=x)
        assert y.dtype == x.dtype


def test_apply_along_last_axis():
    @apply_along_last_axis
    def f(x):
        assert x.ndim == 1
        return x[: len(x) // 2] + np.arange(len(x) // 2)

    for shape in [(10,), (2, 10), (2, 2, 10)]:
        x = np.ones(shape)
        y = f(x)
        xshape = x.shape
        yshape = y.shape
        assert len(xshape) == len(yshape)
        assert xshape[-1] // 2 == yshape[-1]
        y = f(x=x)
        yshape = y.shape
        assert len(xshape) == len(yshape)
        assert xshape[-1] // 2 == yshape[-1]

    # manually expand 1-loop
    x = np.ones((2, 10), dtype=np.float64)
    y = np.empty((2, 5), dtype=np.float64)
    for i in range(len(x)):
        y[i] = f(x[i])
    yhat = f(x)
    assert np.allclose(yhat, y)

    # expand 2-loop
    x = np.ones((2, 2, 10), dtype=np.float64)
    y = np.empty((2, 2, 5), dtype=np.float64)
    for i in range(len(x)):
        for j in range(len(x[i])):
            y[i][j] = f(x[i][j])
    yhat = f(x)
    assert np.allclose(yhat, y)


def test_multiple_decorators():
    @apply_along_last_axis
    @automatic_type_conversion
    def half_vec(x):
        assert x.ndim == 1
        return x[: len(x) // 2]

    for shape in [(10,), (2, 10), (2, 2, 10)]:
        for dtype in [np.float32, np.float16, np.float64]:
            x = np.ones(shape, dtype=dtype)
            y = half_vec(x)
            xshape = x.shape
            yshape = y.shape
            assert len(xshape) == len(yshape)
            assert xshape[-1] // 2 == yshape[-1]
            assert x.dtype == y.dtype
