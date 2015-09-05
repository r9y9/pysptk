# coding: utf-8

import numpy as np
import pysptk
from warnings import warn
from nose.tools import raises


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


def test_lpc2c():
    for src_order in [15, 20, 25, 30]:
        for dst_order in [15, 20, 25, 30]:
            yield __test_conversion_base, pysptk.lpc2c, src_order, dst_order


def test_lpc2lsp():
    for order in [15, 20, 25, 30]:
        yield __test_transform_base, pysptk.lpc2lsp, order

    def __test_invalid_otype(dummy_lpc, otype):
        pysptk.lpc2lsp(dummy_lpc, otype=otype)

    np.random.seed(98765)
    dummy_lpc = pysptk.lpc(np.random.rand(512), 21)

    # invalid otype
    yield raises(ValueError)(__test_invalid_otype), dummy_lpc, 2
    yield raises(ValueError)(__test_invalid_otype), dummy_lpc, 3

    lsp1 = pysptk.lpc2lsp(dummy_lpc, otype=2, fs=16000)
    lsp2 = pysptk.lpc2lsp(dummy_lpc, otype=3, fs=16)
    assert np.allclose(lsp1, lsp2)

    # loggain
    lsp3 = pysptk.lpc2lsp(dummy_lpc,  otype=3, fs=16, loggain=True)
    assert lsp3[0] == np.log(lsp2[0])


def test_lpc2par():
    for order in [15, 20, 25, 30]:
        yield __test_transform_base, pysptk.lpc2par, order


def test_par2lpc():
    for order in [15, 20, 25, 30]:
        yield __test_transform_base, pysptk.par2lpc, order


def test_lsp2sp():
    for order in [15, 20, 25, 30]:
        for fftlen in [256, 512, 1024]:
            yield __test_transform_base, pysptk.lsp2sp, order, fftlen

    def __test_fftlen(fftlen):
        pysptk.lsp2sp(np.ones(20), fftlen)

    for fftlen in [257, 513]:
        yield raises(ValueError)(__test_fftlen), fftlen


def test_mc2b():
    for order in [15, 20, 25, 30]:
        for alpha in [0.35, 0.41, 0.5]:
            yield __test_transform_base, pysptk.mc2b, order, alpha


def test_b2mc():
    for order in [15, 20, 25, 30]:
        for alpha in [0.35, 0.41, 0.5]:
            yield __test_transform_base, pysptk.b2mc, order, alpha


def test_b2c():
    for src_order in [15, 20, 25, 30]:
        for dst_order in [15, 20, 25, 30]:
            for alpha in [0.35, 0.41, 0.5]:
                yield __test_conversion_base, pysptk.b2c, src_order, dst_order, alpha


def test_c2acr():
    for src_order in [15, 20, 25, 30]:
        for dst_order in [15, 20, 25, 30]:
            for fftlen in [256, 512, 1024]:
                yield __test_transform_base, pysptk.b2c, src_order, dst_order, fftlen

    def __test_fftlen(fftlen):
        pysptk.c2acr(np.ones(20), 19, fftlen)

    for fftlen in [257, 513]:
        yield raises(ValueError)(__test_fftlen), fftlen


def test_c2ir():
    for order in [15, 20, 25, 30]:
        for length in [256, 512, 1024]:
            yield __test_conversion_base, pysptk.c2ir, order, length


def test_ic2ir():
    for length in [256, 512, 1024]:
        for order in [15, 20, 25, 30]:
            yield __test_conversion_base, pysptk.ic2ir, length, order


def test_ic2ir_invertibility():
    def __test(order, length):
        np.random.seed(98765)
        dummy_ceps = np.random.rand(order + 1)
        ir = pysptk.c2ir(dummy_ceps, length)
        c = pysptk.ic2ir(ir, order)
        assert np.allclose(c, dummy_ceps)

    for order in [15, 20, 25, 30]:
        for length in [256, 512, 1024]:
            yield __test, order, length


def test_c2ndps():
    for order in [15, 20, 25, 30]:
        for fftlen in [256, 512, 1024]:
            yield __test_conversion_base, pysptk.c2ndps, order, fftlen

    # invalid fftlen
    for fftlen in [255, 257]:
        yield raises(ValueError)(__test_conversion_base), pysptk.c2ndps, 20, fftlen


def test_ndps2c():
    for fftlen in [256, 512, 1024]:
        for order in [15, 20, 25, 30]:
            yield __test_conversion_base, pysptk.ndps2c, (fftlen >> 1), order

    def __test(length):
        pysptk.ndps2c(np.ones(length), 20)

    # invalid fftlen
    for fftlen in [255, 257]:
        yield raises(ValueError)(__test), (fftlen >> 1)


def test_gc2gc():
    def __test(src_order, src_gamma, dst_order, dst_gamma):
        np.random.seed(98765)
        src = np.random.rand(src_order + 1)
        dst = pysptk.gc2gc(src, src_gamma, dst_order, dst_gamma)
        assert np.all(np.isfinite(dst))

    for src_order in [15, 20, 25, 30]:
        for src_gamma in [-1.0, -0.5, 0.0]:
            for dst_order in [15, 20, 25, 30]:
                for dst_gamma in [-1.0, -0.5, 0.0]:
                    yield __test, src_order, src_gamma, dst_order, dst_gamma

    # invalid gamma
    yield raises(ValueError)(__test), 20, 0.0, 20, 0.1
    yield raises(ValueError)(__test), 20, 0.1, 20, 0.0


def test_gnorm():
    for order in [15, 20, 25, 30]:
        for gamma in [-1.0, -0.5, 0.0]:
            yield __test_transform_base, pysptk.gnorm, order, gamma

    # invalid gamma
    yield raises(ValueError)(__test_transform_base), pysptk.gnorm, 20, 0.1


def test_ignorm():
    for order in [15, 20, 25, 30]:
        for gamma in [-1.0, -0.5, 0.0]:
            yield __test_transform_base, pysptk.ignorm, order, gamma

    # invalid gamma
    yield raises(ValueError)(__test_transform_base), pysptk.ignorm, 20, 0.1


def test_freqt():
    for src_order in [15, 20, 25, 30]:
        for dst_order in [15, 20, 25, 30]:
            for alpha in [0.35, 0.41, 0.5]:
                yield __test_conversion_base, pysptk.freqt, src_order, dst_order, alpha
                yield __test_conversion_base, pysptk.frqtr, src_order, dst_order, alpha


def test_mgc2mgc():
    def __test(src_order, src_alpha, src_gamma, dst_order, dst_alpha, dst_gamma):
        np.random.seed(98765)
        src = np.random.rand(src_order + 1)
        dst = pysptk.mgc2mgc(src, src_alpha, src_gamma,
                             dst_order, dst_alpha, dst_gamma)
        assert np.all(np.isfinite(dst))

    for src_order in [15, 20, 25, 30]:
        for src_alpha in [0.35, 0.41, 0.5]:
            for src_gamma in [-1.0, -0.5, 0.0]:
                for dst_order in [15, 20, 25, 30]:
                    for dst_alpha in [0.35, 0.41, 0.5]:
                        for dst_gamma in [-1.0, -0.5, 0.0]:
                            yield __test, src_order, src_alpha, src_gamma, dst_order, dst_alpha, dst_gamma

    # invalid gamma
    yield raises(ValueError)(__test), 20, 0.0, 0.1, 20, 0.0, 0.0
    yield raises(ValueError)(__test), 20, 0.0, 0.0, 20, 0.0, 0.1


def test_mgc2sp():
    def __test(order, alpha, gamma, fftlen):
        np.random.seed(98765)
        src = np.random.rand(order + 1)
        dst = pysptk.mgc2sp(src, alpha, gamma, fftlen)
        assert len(dst) == (fftlen >> 1) + 1
        assert np.all(np.isfinite(dst))

    for order in [15, 20, 25, 30]:
        for alpha in [0.35, 0.41, 0.5]:
            for gamma in [-1.0, -0.5, 0.0]:
                for fftlen in [256, 512, 1024]:
                    yield __test, order, alpha, gamma, fftlen


def test_mgclsp2sp():
    def __test(order, alpha, gamma, fftlen):
        np.random.seed(98765)
        src = np.random.rand(order + 1)
        dst = pysptk.mgclsp2sp(src, alpha, gamma, fftlen)
        assert len(dst) == (fftlen >> 1) + 1
        assert np.all(np.isfinite(dst))

    # TODO
    warn("Inf/-Inf wiil happens when gamma = 0.0")
    for order in [15, 20, 25, 30]:
        for alpha in [0.35, 0.41, 0.5]:
            for gamma in [-1.0, -0.5]:
                for fftlen in [256, 512, 1024]:
                    yield __test, order, alpha, gamma, fftlen

    # invalid gamma
    yield raises(ValueError)(__test), 20, 0.0, 0.1, 256

    # invalid fftlen
    yield raises(ValueError)(__test), 20, 0.0, -0.1, 255
    yield raises(ValueError)(__test), 20, 0.0, -0.1, 257


def test_mgc2b():
    for order in [15, 20, 25, 30]:
        for alpha in [0.0, 0.35, 0.41]:
            for gamma in [-1.0, -0.5, 0.0]:
                yield __test_transform_base, pysptk.mgc2b, order, alpha, gamma
