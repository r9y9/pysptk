# coding: utf-8

import numpy as np
import six
import pysptk
from warnings import warn
from nose.tools import raises


def windowed_dummy_data(N):
    np.random.seed(98765)
    return pysptk.hanning(N) * np.random.randn(N)


def test_assert_gamma():
    def __test(gamma):
        pysptk.assert_gamma(gamma)

    for gamma in [-2.0, 0.1]:
        yield raises(ValueError)(__test), gamma


def test_assert_pade():
    def __test(pade):
        pysptk.assert_pade(pade)

    for pade in [3, 6]:
        yield raises(ValueError)(__test), pade


def test_assert_fftlen():
    def __test(fftlen):
        pysptk.assert_fftlen(fftlen)

    for fftlen in [255, 257]:
        yield raises(ValueError)(__test), fftlen


def test_agexp():
    assert pysptk.agexp(1, 1, 1) == 5.0
    assert pysptk.agexp(1, 2, 3) == 18.0


def test_gexp():
    assert pysptk.gexp(1, 1) == 2.0
    assert pysptk.gexp(2, 4) == 3.0


def test_glog():
    assert pysptk.glog(1, 2) == 1.0
    assert pysptk.glog(2, 3) == 4.0


def test_mseq():
    for i in six.moves.range(0, 100):
        assert np.isfinite(pysptk.mseq())


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


def test_swipe():

    def __test(x, fs, hopsize, otype):
        f0 = pysptk.swipe(x, fs, hopsize, otype=otype)
        assert np.all(np.isfinite(f0))

    np.random.seed(98765)
    fs = 16000
    hopsize = 80
    x = np.random.rand(16000)

    for otype in six.moves.range(0, 3):
        yield __test, x, fs, hopsize, otype

    # unsupported otype
    yield raises(ValueError)(__test), x, fs, hopsize, -1
    yield raises(ValueError)(__test), x, fs, hopsize, 3


def test_windows():
    def __test(f, N, normalize):
        w = f(N, normalize)
        assert np.all(np.isfinite(w))

    from pysptk import blackman, hanning, hamming
    from pysptk import bartlett, trapezoid, rectangular

    for f in [blackman, hanning, hamming, bartlett, trapezoid, rectangular]:
        for n in [16, 128, 256, 1024, 2048, 4096]:
            yield __test, f, n, 1

    # unsupported normalize flags
    for f in [blackman, hanning, hamming, bartlett, trapezoid, rectangular]:
        yield raises(ValueError)(__test), f, 256, -1
        yield raises(ValueError)(__test), f, 256, 3


def test_mcep():
    x = windowed_dummy_data(1024)

    def __test(order, alpha):
        mc = pysptk.mcep(x, order, alpha)
        assert np.all(np.isfinite(mc))

    for order in [15, 20, 25]:
        for alpha in [0.0, 0.35, 0.41]:
            yield __test, order, alpha


def test_gcep():
    x = windowed_dummy_data(1024)

    def __test(order, gamma):
        gc = pysptk.gcep(x, order, gamma)
        assert np.all(np.isfinite(gc))

    for order in [15, 20, 25]:
        for gamma in [0.0, -0.25, -0.5]:
            yield __test, order, gamma


def test_mgcep():
    x = windowed_dummy_data(1024)

    def __test(order, alpha, gamma):
        mgc = pysptk.mgcep(x, order, alpha, gamma)
        assert np.all(np.isfinite(mgc))

    for order in [15, 20, 25]:
        for alpha in [0.0, 0.35, 0.41]:
            for gamma in [0.0, -0.25, -0.5]:
                yield __test, order, alpha, gamma


def test_uels():
    x = windowed_dummy_data(1024)

    def __test(order):
        c = pysptk.uels(x, order)
        assert np.all(np.isfinite(c))

    for order in [15, 20, 25]:
        yield __test, order


def test_fftcep():
    x = windowed_dummy_data(1024)
    logsp = np.log(np.abs(np.fft.rfft(x)) + 1.0e-6)

    def __test(order):
        c = pysptk.fftcep(logsp, order)
        assert np.all(np.isfinite(c))

    warn("TODO: fix memory corruption in fftcep")
    # for order in [15, 20, 25]:
    #    yield __test, order


def test_lpc():
    x = windowed_dummy_data(1024)

    def __test(order):
        a = pysptk.lpc(x, order)
        assert np.all(np.isfinite(a))

    for order in [15, 20, 25]:
        yield __test, order


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


def test_lmadf():
    for order in [20, 25, 30]:
        for pd in [4, 5]:
            delay = pysptk.lmadf_delay(order, pd)
            yield __test_filt_base, pysptk.lmadf, order, delay, pd


def test_lspdf():
    for order in [20, 25, 30]:
        delay = pysptk.lspdf_delay(order)
        yield __test_filt_base, pysptk.lspdf, order, delay


def test_ltcdf():
    for order in [20, 25, 30]:
        delay = pysptk.ltcdf_delay(order)
        yield __test_filt_base, pysptk.ltcdf, order, delay


def test_glsadf():
    for order in [20, 25, 30]:
        for stage in six.moves.range(1, 7):
            delay = pysptk.glsadf_delay(order, stage)
            yield __test_filt_base, pysptk.glsadf, order, delay, stage


def test_mlsadf():
    for order in [20, 25, 30]:
        for alpha in [0.0, 0.35, 0.5]:
            for pd in [4, 5]:
                delay = pysptk.mlsadf_delay(order, pd)
                yield __test_filt_base, pysptk.mlsadf, order, delay, alpha, pd


def test_mglsadf():
    for order in [20, 25, 30]:
        for alpha in [0.0, 0.35, 0.5]:
            for stage in six.moves.range(1, 7):
                delay = pysptk.mglsadf_delay(order, stage)
                yield __test_filt_base, pysptk.mglsadf, order, delay, alpha, stage


def test_phidf():
    def __test(order, alpha):
        np.random.seed(98765)
        dummy_input = np.random.rand(64)
        delay = np.zeros(order + 1)
        for x in dummy_input:
            pysptk.phidf(x, order, alpha, delay)
            assert np.all(np.isfinite(delay))

    for order in [15, 20, 25, 30]:
        for alpha in [0.35, 0.41, 0.5]:
            yield __test, order, alpha


def test_lspcheck():
    def __test(order):
        np.random.seed(98765)
        lsp = np.random.rand(order + 1)
        pysptk.lspcheck(lsp)
        # TODO: valid check

    for order in [15, 20, 25, 30]:
        yield __test, order
