from warnings import warn

import numpy as np
import pysptk
import pytest


def windowed_dummy_data(N):
    np.random.seed(98765)
    return pysptk.hanning(N) * np.random.randn(N)


def __test_conversion_base(f, src_order, dst_order, *args, **kwargs):
    np.random.seed(98765)
    src = np.random.rand(src_order + 1)
    dst = f(src, dst_order, *args, **kwargs)

    assert np.all(np.isfinite(dst))


def __test_transform_base(f, order, *args, **kwargs):
    np.random.seed(98765)
    src = np.random.rand(order + 1)
    dst = f(src, *args, **kwargs)

    assert np.all(np.isfinite(dst))


@pytest.mark.parametrize("src_order", [15, 20, 25, 30])
@pytest.mark.parametrize("dst_order", [15, 20, 25, 30])
def test_lpc2c(src_order, dst_order):
    __test_conversion_base(pysptk.lpc2c, src_order, dst_order)


def test_lpc2lsp():
    # for order in [15, 20, 25, 30]:
    #    yield __test_transform_base, pysptk.lpc2lsp, order

    def __test_invalid_otype(dummy_lpc, otype):
        pysptk.lpc2lsp(dummy_lpc, otype=otype)

    np.random.seed(98765)
    dummy_lpc = pysptk.lpc(np.random.rand(512), 21)

    # invalid otype
    with pytest.raises(ValueError):
        __test_invalid_otype(dummy_lpc, 2)
    with pytest.raises(ValueError):
        __test_invalid_otype(dummy_lpc, 3)

    lsp1 = pysptk.lpc2lsp(dummy_lpc, otype=2, fs=16000)
    lsp2 = pysptk.lpc2lsp(dummy_lpc, otype=3, fs=16)
    assert np.allclose(lsp1, lsp2)

    # loggain
    lsp3 = pysptk.lpc2lsp(dummy_lpc, otype=3, fs=16, loggain=True)
    assert lsp3[0] == np.log(lsp2[0])


@pytest.mark.parametrize("order", [15, 20, 25, 30])
def test_lpc2par(order):
    __test_transform_base(pysptk.lpc2par, order)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
def test_par2lpc(order):
    __test_transform_base(pysptk.par2lpc, order)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("fftlen", [256, 512, 1024])
def test_lsp2sp(order, fftlen):
    __test_transform_base(pysptk.lsp2sp, order, fftlen)


def test_lsp2sp_fftlen():
    def __test_fftlen(fftlen):
        pysptk.lsp2sp(np.ones(20), fftlen)

    with pytest.raises(ValueError):
        __test_fftlen(257)
    with pytest.raises(ValueError):
        __test_fftlen(513)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("alpha", [0.35, 0.41, 0.5])
def test_mc2b(order, alpha):
    __test_transform_base(pysptk.mc2b, order, alpha)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("alpha", [0.35, 0.41, 0.5])
def test_b2mc(order, alpha):
    __test_transform_base(pysptk.b2mc, order, alpha)


@pytest.mark.parametrize("src_order", [15, 20, 25, 30])
@pytest.mark.parametrize("dst_order", [15, 20, 25, 30])
@pytest.mark.parametrize("alpha", [0.35, 0.41, 0.5])
def test_b2c(src_order, dst_order, alpha):
    __test_conversion_base(pysptk.b2c, src_order, dst_order, alpha)


@pytest.mark.parametrize("src_order", [15, 20, 25, 30])
@pytest.mark.parametrize("dst_order", [15, 20, 25, 30])
@pytest.mark.parametrize("fftlen", [256, 512, 1024])
def test_c2acr(src_order, dst_order, fftlen):
    __test_transform_base(pysptk.b2c, src_order, dst_order, fftlen)


def test_c2acr_fftlen():
    def __test_fftlen(fftlen):
        pysptk.c2acr(np.ones(20), 19, fftlen)

    with pytest.raises(ValueError):
        __test_fftlen(257)
    with pytest.raises(ValueError):
        __test_fftlen(513)
    with pytest.raises(ValueError):
        __test_fftlen(16)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("length", [256, 512, 1024])
def test_c2ir(order, length):
    __test_conversion_base(pysptk.c2ir, order, length)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("length", [256, 512, 1024])
def test_ic2ir(order, length):
    __test_conversion_base(pysptk.ic2ir, length, order)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("length", [256, 512, 1024])
def test_ic2ir_invertibility(order, length):
    np.random.seed(98765)
    dummy_ceps = np.random.rand(order + 1)
    ir = pysptk.c2ir(dummy_ceps, length)
    c = pysptk.ic2ir(ir, order)
    assert np.allclose(c, dummy_ceps)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("fftlen", [256, 512, 1024])
def test_c2ndps(order, fftlen):
    __test_conversion_base(pysptk.c2ndps, order, fftlen)


def test_c2ndps_fftlen():
    # invalid fftlen
    with pytest.raises(ValueError):
        __test_conversion_base(pysptk.c2ndps, 20, 255)
    with pytest.raises(ValueError):
        __test_conversion_base(pysptk.c2ndps, 20, 257)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("fftlen", [256, 512, 1024])
def test_ndps2c(order, fftlen):
    __test_conversion_base(pysptk.ndps2c, (fftlen >> 1), order)


def test_ndps2c_fftlen():
    def __test(length):
        pysptk.ndps2c(np.ones(length), 20)

    # invalid fftlen
    with pytest.raises(ValueError):
        __test(255 >> 1)
    with pytest.raises(ValueError):
        __test(257 >> 1)


@pytest.mark.parametrize("src_order", [15, 20, 25, 30])
@pytest.mark.parametrize("src_gamma", [-1.0, -0.5, 0.0])
@pytest.mark.parametrize("dst_order", [15, 20, 25, 30])
@pytest.mark.parametrize("dst_gamma", [-1.0, -0.5, 0.0])
def test_gc2gc(src_order, src_gamma, dst_order, dst_gamma):
    np.random.seed(98765)
    src = np.random.rand(src_order + 1)
    dst = pysptk.gc2gc(src, src_gamma, dst_order, dst_gamma)
    assert np.all(np.isfinite(dst))


def test_gc2gc_gamma():
    def __test(src_order, src_gamma, dst_order, dst_gamma):
        np.random.seed(98765)
        src = np.random.rand(src_order + 1)
        dst = pysptk.gc2gc(src, src_gamma, dst_order, dst_gamma)
        assert np.all(np.isfinite(dst))

    # invalid gamma
    with pytest.raises(ValueError):
        __test(20, 0.0, 20, 0.1)
    with pytest.raises(ValueError):
        __test(20, 0.1, 20, 0.0)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("gamma", [-1.0, -0.5, 0.0])
def test_gnorm(order, gamma):
    __test_transform_base(pysptk.gnorm, order, gamma)


def test_gnorm_gamma():
    # invalid gamma
    with pytest.raises(ValueError):
        __test_transform_base(pysptk.gnorm, 20, 0.1)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("gamma", [-1.0, -0.5, 0.0])
def test_ignorm(order, gamma):
    __test_transform_base(pysptk.ignorm, order, gamma)


def test_ignorm_gamma():
    # invalid gamma
    with pytest.raises(ValueError):
        __test_transform_base(pysptk.ignorm, 20, 0.1)


@pytest.mark.parametrize("src_order", [15, 20, 25, 30])
@pytest.mark.parametrize("dst_order", [15, 20, 25, 30])
@pytest.mark.parametrize("alpha", [0.35, 0.41, 0.5])
def test_freqt(src_order, dst_order, alpha):
    __test_conversion_base(pysptk.freqt, src_order, dst_order, alpha)


@pytest.mark.parametrize("src_order", [15, 20, 25, 30])
@pytest.mark.parametrize("src_alpha", [0.35, 0.41, 0.5])
@pytest.mark.parametrize("src_gamma", [-1.0, -0.5, 0.0])
@pytest.mark.parametrize("dst_order", [15, 20, 25, 30])
@pytest.mark.parametrize("dst_alpha", [0.35, 0.41, 0.5])
@pytest.mark.parametrize("dst_gamma", [-1.0, -0.5, 0.0])
def test_mgc2mgc(src_order, src_alpha, src_gamma, dst_order, dst_alpha, dst_gamma):
    np.random.seed(98765)
    src = np.random.rand(src_order + 1)
    dst = pysptk.mgc2mgc(src, src_alpha, src_gamma, dst_order, dst_alpha, dst_gamma)
    assert np.all(np.isfinite(dst))


def test_mgc2mgc_gamma():
    def __test(src_order, src_alpha, src_gamma, dst_order, dst_alpha, dst_gamma):
        np.random.seed(98765)
        src = np.random.rand(src_order + 1)
        dst = pysptk.mgc2mgc(src, src_alpha, src_gamma, dst_order, dst_alpha, dst_gamma)
        assert np.all(np.isfinite(dst))

    # invalid gamma
    with pytest.raises(ValueError):
        __test(20, 0.0, 0.1, 20, 0.0, 0.0)
    with pytest.raises(ValueError):
        __test(20, 0.0, 0.0, 20, 0.0, 0.1)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("alpha", [0.35, 0.41, 0.5])
@pytest.mark.parametrize("gamma", [-1.0, -0.5, 0.0])
@pytest.mark.parametrize("fftlen", [256, 512, 1024])
def test_mgc2sp(order, alpha, gamma, fftlen):
    np.random.seed(98765)
    src = np.random.rand(order + 1)
    dst = pysptk.mgc2sp(src, alpha, gamma, fftlen)
    assert len(dst) == (fftlen >> 1) + 1
    assert np.all(np.isfinite(dst))


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("alpha", [0.35, 0.41, 0.5])
@pytest.mark.parametrize("gamma", [-1.0, -0.5, 0.0])
@pytest.mark.parametrize("fftlen", [256, 512, 1024])
def test_mgclsp2sp(order, alpha, gamma, fftlen):
    def __test(order, alpha, gamma, fftlen):
        np.random.seed(98765)
        src = np.random.rand(order + 1)
        dst = pysptk.mgclsp2sp(src, alpha, gamma, fftlen)
        assert len(dst) == (fftlen >> 1) + 1
        assert np.all(np.isfinite(dst))

    # TODO
    if gamma == 0.0:
        warn("Inf/-Inf wiil happens when gamma = 0.0")
        return

    __test(order, alpha, gamma, fftlen)


def test_mgclsp2sp_corner_case():
    def __test(order, alpha, gamma, fftlen):
        np.random.seed(98765)
        src = np.random.rand(order + 1)
        pysptk.mgclsp2sp(src, alpha, gamma, fftlen)

    # invalid gamma
    with pytest.raises(ValueError):
        __test(20, 0.0, 0.1, 256)

    # invalid fftlen
    with pytest.raises(ValueError):
        __test(20, 0.0, -0.1, 255)
    with pytest.raises(ValueError):
        __test(20, 0.0, -0.1, 257)


@pytest.mark.parametrize("order", [15, 20, 25, 30])
@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.41])
@pytest.mark.parametrize("gamma", [-1.0, -0.5, 0.0])
def test_mgc2b(order, alpha, gamma):
    __test_transform_base(pysptk.mgc2b, order, alpha, gamma)


@pytest.mark.parametrize("order", [10, 20, 40, 50, 60])
@pytest.mark.parametrize("alpha", [0.35, 0.41])
@pytest.mark.parametrize("fftlen", [512, 1024, 2048])
def test_sp2mc(order, alpha, fftlen):
    np.random.seed(98765)
    sp = np.random.rand(int(fftlen // 2 + 1))
    mc = pysptk.sp2mc(sp, order, alpha)
    approx_sp = pysptk.mc2sp(mc, alpha, fftlen)
    # TODO: tolerance should be more carefully chosen
    assert np.allclose(sp, approx_sp, atol=0.9)


def test_mc2e():
    x = windowed_dummy_data(1024)
    mc = pysptk.mcep(x)
    assert pysptk.mc2e(mc) > 0
