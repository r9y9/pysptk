# coding: utf-8

from __future__ import print_function, absolute_import

import numpy as np
import pysptk
from pysptk.synthesis import Synthesizer

from nose.tools import raises
from warnings import warn


def __dummy_source():
    np.random.seed(98765)
    return np.random.randn(2**14)


def __dummy_windowed_frames(source, frame_len=512, hopsize=80):
    np.random.seed(98765)
    n_frames = int(len(source) / hopsize) + 1
    windowed = np.random.randn(n_frames, frame_len) * pysptk.blackman(frame_len)
    return 0.5 * 32768.0 * windowed


def test_LMADF():
    from pysptk.synthesis import LMADF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        b = pysptk.mcep(windowed, filt.order, 0.0)

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

    def __test(order, pd):
        __test_synthesis(LMADF(order, pd=pd))

    for pd in [4, 5, 6, 7]:
        for order in [20, 25]:
            yield __test, order, pd

    def __test_invalid_pade(pd):
        LMADF(20, pd=pd)

    yield raises(ValueError)(__test_invalid_pade), 3
    yield raises(ValueError)(__test_invalid_pade), 8


def test_MLSADF():
    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        mc = pysptk.mcep(windowed, filt.order, filt.alpha)
        b = pysptk.mc2b(mc, filt.alpha)

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

        # transpose
        synthesizer = Synthesizer(filt, hopsize, transpose=True)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

    from pysptk.synthesis import MLSADF

    def __test(order, alpha, pd):
        __test_synthesis(MLSADF(order, alpha, pd=pd))

    for pd in [4, 5, 6, 7]:
        for order in [20, 25]:
            for alpha in [0.0, 0.41]:
                yield __test, order, alpha, pd

    def __test_invalid_pade(pd):
        MLSADF(20, pd=pd)

    yield raises(ValueError)(__test_invalid_pade), 3
    yield raises(ValueError)(__test_invalid_pade), 8


def test_GLSADF():
    from pysptk.synthesis import GLSADF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        gamma = -1.0 / filt.stage
        mgc = pysptk.mgcep(windowed, filt.order, 0.0, gamma)
        b = pysptk.mgc2b(mgc, 0.0, gamma)

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

        # transpose
        synthesizer = Synthesizer(filt, hopsize, transpose=True)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

    def __test(order, stage):
        __test_synthesis(GLSADF(order, stage))

    for order in [20, 25]:
        for stage in [2, 5, 10]:
            yield __test, order, stage

    def __test_invalid_stage(stage):
        GLSADF(20, stage=stage)

    yield raises(ValueError)(__test_invalid_stage), -1
    yield raises(ValueError)(__test_invalid_stage), 0


def test_MGLSADF():
    from pysptk.synthesis import MGLSADF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        gamma = -1.0 / filt.stage
        mgc = pysptk.mgcep(windowed, filt.order, filt.alpha, gamma)
        b = pysptk.mgc2b(mgc, filt.alpha, gamma)

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

        # transpose
        synthesizer = Synthesizer(filt, hopsize, transpose=True)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

    def __test(order, alpha, stage):
        __test_synthesis(MGLSADF(order, alpha, stage))

    for order in [20, 25]:
        for alpha in [0.0, 0.41]:
            for stage in [2, 5, 10]:
                yield __test, order, alpha, stage

    def __test_invalid_stage(stage):
        MGLSADF(20, stage=stage)

    yield raises(ValueError)(__test_invalid_stage), -1
    yield raises(ValueError)(__test_invalid_stage), 0


def test_AllZeroDF():
    from pysptk.synthesis import AllZeroDF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        lpc = pysptk.lpc(windowed, filt.order)
        lpc[:, 0] = 0
        b = -lpc

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

        # transpose
        synthesizer = Synthesizer(filt, hopsize, transpose=True)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))


def test_AllPoleDF():
    from pysptk.synthesis import AllPoleDF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        lpc = pysptk.lpc(windowed, filt.order)

        # make sure lpc has loggain
        lpc[:, 0] = np.log(lpc[:, 0])

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, lpc)
        assert np.all(np.isfinite(y))

        # transpose
        synthesizer = Synthesizer(filt, hopsize, transpose=True)
        y = synthesizer.synthesis(source, lpc)
        assert np.all(np.isfinite(y))

    def __test_synthesis_levdur(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        c = pysptk.mcep(windowed, filt.order)
        a = pysptk.c2acr(c)
        lpc = pysptk.levdur(a)
        lpc2 = pysptk.levdur(a, use_scipy=True)
        assert np.allclose(lpc, lpc2)

        # make sure lpc has loggain
        lpc[:, 0] = np.log(lpc[:, 0])

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, lpc)
        assert np.all(np.isfinite(y))

    def __test(order):
        __test_synthesis(AllPoleDF(order))
        __test_synthesis_levdur(AllPoleDF(order))

    for order in [20, 25]:
        yield __test, order


def test_AllPoleLatticeDF():
    from pysptk.synthesis import AllPoleLatticeDF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        lpc = pysptk.lpc(windowed, filt.order)
        par = pysptk.lpc2par(lpc)

        # make sure par has loggain
        par[:, 0] = np.log(par[:, 0])

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, par)
        assert np.all(np.isfinite(y))

    def __test(order):
        __test_synthesis(AllPoleLatticeDF(order))

    for order in [20, 25]:
        yield __test, order


def test_LSPDF():
    from pysptk.synthesis import LSPDF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        lpc = pysptk.lpc(windowed, filt.order)
        lsp = pysptk.lpc2lsp(lpc)
        # make sure lsp has loggain
        lsp[:, 0] = np.log(lsp[:, 0])

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, lsp)
        assert np.all(np.isfinite(y))

    def __test(order):
        __test_synthesis(LSPDF(order))

    for order in [20, 25]:
        yield __test, order
