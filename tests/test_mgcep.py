import numpy as np
import pysptk
import pytest


def windowed_dummy_data(N):
    np.random.seed(98765)
    return pysptk.hanning(N) * np.random.randn(N)


def windowed_dummy_frames(T, N, dtype=np.float64):
    np.random.seed(98765)
    frames = pysptk.hanning(N) * np.random.randn(T, N)
    return frames.astype(np.float64)


@pytest.mark.parametrize("order", [15, 20, 25])
@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.41])
def test_mcep(order, alpha):
    x = windowed_dummy_data(1024)

    def __test(order, alpha):
        mc = pysptk.mcep(x, order, alpha)
        assert np.all(np.isfinite(mc))

    __test(order, alpha)


def test_mcep_broadcast():
    # Test for broadcasting
    def __test_broadcast(dtype):
        frames = windowed_dummy_frames(100, 512, dtype=dtype)
        mc = pysptk.mcep(frames, 20, 0.41)
        assert np.all(np.isfinite(mc))
        assert frames.shape[0] == mc.shape[0]

    for dtype in [np.float16, np.float32, np.float64]:
        __test_broadcast(dtype)


def test_mcep_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_itype(itype=0):
        pysptk.mcep(x, itype=itype)

    with pytest.raises(ValueError):
        __test_itype(-1)
    with pytest.raises(ValueError):
        __test_itype(5)

    def __test_eps(etype=0, eps=0.0):
        pysptk.mcep(x, etype=etype, eps=eps)

    with pytest.raises(ValueError):
        __test_eps(0, -1.0)
    with pytest.raises(ValueError):
        __test_eps(-1)
    with pytest.raises(ValueError):
        __test_eps(-3)
    with pytest.raises(ValueError):
        __test_eps(1, -1.0)
    with pytest.raises(ValueError):
        __test_eps(2, 0.0)
    with pytest.raises(ValueError):
        __test_eps(2, 1.0)

    def __test_min_det(min_det):
        pysptk.mcep(x, min_det=min_det)

    with pytest.raises(ValueError):
        __test_min_det(-1.0)


def test_mcep_failure():
    with pytest.raises(RuntimeError):
        pysptk.mcep(np.ones(256), 40, 0.41)


@pytest.mark.parametrize("order", [15, 20, 25])
@pytest.mark.parametrize("gamma", [0.0, -0.25, -0.5])
def test_gcep(order, gamma):
    x = windowed_dummy_data(1024)
    gc = pysptk.gcep(x, order, gamma)
    assert np.all(np.isfinite(gc))


def test_gcep_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_gamma(gamma):
        pysptk.gcep(x, gamma=gamma)

    with pytest.raises(ValueError):
        __test_gamma(0.1)
    with pytest.raises(ValueError):
        __test_gamma(-2.1)

    def __test_itype(itype=0):
        pysptk.gcep(x, itype=itype)

    with pytest.raises(ValueError):
        __test_itype(-1)
    with pytest.raises(ValueError):
        __test_itype(5)

    def __test_eps(etype=0, eps=0.0):
        pysptk.gcep(x, etype=etype, eps=eps)

    with pytest.raises(ValueError):
        __test_eps(0, -1.0)
    with pytest.raises(ValueError):
        __test_eps(-1)
    with pytest.raises(ValueError):
        __test_eps(-3)
    with pytest.raises(ValueError):
        __test_eps(1, -1.0)
    with pytest.raises(ValueError):
        __test_eps(2, -1.0)

    def __test_min_det(min_det):
        pysptk.gcep(x, min_det=min_det)

    with pytest.raises(ValueError):
        __test_min_det(-1.0)


def test_gcep_failure():
    with pytest.raises(RuntimeError):
        pysptk.gcep(np.ones(256), 40, 0.0)


@pytest.mark.parametrize("order", [15, 20, 25])
@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.41])
@pytest.mark.parametrize("gamma", [-0.5, -0.25, 0.0])
def test_mgcep(order, alpha, gamma):
    x = windowed_dummy_data(1024)

    mgc = pysptk.mgcep(x, order, alpha, gamma)
    assert np.all(np.isfinite(mgc))


def test_mgcep_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_gamma(gamma):
        pysptk.mgcep(x, gamma=gamma)

    with pytest.raises(ValueError):
        __test_gamma(0.1)
    with pytest.raises(ValueError):
        __test_gamma(-2.1)

    def __test_itype(itype=0):
        pysptk.mgcep(x, itype=itype)

    with pytest.raises(ValueError):
        __test_itype(-1)
    with pytest.raises(ValueError):
        __test_itype(5)

    def __test_eps(etype=0, eps=0.0):
        pysptk.mgcep(x, etype=etype, eps=eps)

    with pytest.raises(ValueError):
        __test_eps(0, -1.0)
    with pytest.raises(ValueError):
        __test_eps(-1)
    with pytest.raises(ValueError):
        __test_eps(-3)
    with pytest.raises(ValueError):
        __test_eps(1, -1.0)
    with pytest.raises(ValueError):
        __test_eps(2, -1.0)

    def __test_min_det(min_det):
        pysptk.mgcep(x, min_det=min_det)

    with pytest.raises(ValueError):
        __test_min_det(-1.0)

    def __test_otype(otype=0):
        pysptk.mgcep(x, otype=otype)

    with pytest.raises(ValueError):
        __test_otype(-1)
    with pytest.raises(ValueError):
        __test_otype(6)


def test_mgcep_failure():
    with pytest.raises(RuntimeError):
        pysptk.mgcep(np.ones(256))


@pytest.mark.parametrize("order", [15, 20, 25])
def test_uels(order):
    x = windowed_dummy_data(1024)

    c = pysptk.uels(x, order)
    assert np.all(np.isfinite(c))


def test_uels_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_itype(itype=0):
        pysptk.uels(x, itype=itype)

    with pytest.raises(ValueError):
        __test_itype(-1)
    with pytest.raises(ValueError):
        __test_itype(5)

    def __test_eps(etype=0, eps=0.0):
        pysptk.uels(x, etype=etype, eps=eps)

    with pytest.raises(ValueError):
        __test_eps(0, -1.0)
    with pytest.raises(ValueError):
        __test_eps(-1)
    with pytest.raises(ValueError):
        __test_eps(-3)
    with pytest.raises(ValueError):
        __test_eps(1, -1.0)
    with pytest.raises(ValueError):
        __test_eps(2, -1.0)


def test_uels_failure():
    with pytest.raises(RuntimeError):
        pysptk.uels(np.ones(256), 40)


@pytest.mark.parametrize("order", [15, 20, 25])
def test_fftcep(order):
    x = windowed_dummy_data(1024)
    logsp = np.log(np.abs(np.fft.rfft(x)) + 1.0e-6)

    c = pysptk.fftcep(logsp, order)
    assert np.all(np.isfinite(c))


@pytest.mark.parametrize("order", [15, 20, 25])
def test_lpc(order):
    x = windowed_dummy_data(1024)

    a1 = pysptk.lpc(x, order, use_scipy=False)
    a2 = pysptk.lpc(x, order, use_scipy=True)
    a3 = pysptk.levdur(pysptk.acorr(x, order), use_scipy=False)
    a4 = pysptk.levdur(pysptk.acorr(x, order), use_scipy=True)

    assert np.all(np.isfinite(a1))
    assert np.allclose(a1, a2)
    assert np.allclose(a1, a3)
    assert np.allclose(a1, a4)


def test_lpc_invalid_args():
    x = windowed_dummy_data(1024)

    def __test_min_det(min_det):
        pysptk.lpc(x, min_det=min_det, use_scipy=False)

    with pytest.raises(ValueError):
        __test_min_det(-1.0)


def test_lpc_failure():
    with pytest.raises(RuntimeError):
        pysptk.lpc(np.zeros(256), 40, use_scipy=False)
