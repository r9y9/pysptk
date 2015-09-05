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
        b = np.apply_along_axis(
            pysptk.mcep, 1, windowed, filt.order, 0.0)

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

    def __test(order):
        __test_synthesis(LMADF(order))

    for order in [20, 25]:
        yield __test, order

    def __test_invalid_pade(pd):
        LMADF(20, pd=pd)

    yield raises(ValueError)(__test_invalid_pade), 3
    yield raises(ValueError)(__test_invalid_pade), 6


def test_MLSADF():
    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        mc = np.apply_along_axis(
            pysptk.mcep, 1, windowed, filt.order, filt.alpha)
        b = np.apply_along_axis(pysptk.mc2b, 1, mc, filt.alpha)

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

    from pysptk.synthesis import MLSADF

    def __test(order, alpha):
        __test_synthesis(MLSADF(order, alpha))

    for order in [20, 25]:
        for alpha in [0.0, 0.41]:
            yield __test, order, alpha

    def __test_invalid_pade(pd):
        MLSADF(20, pd=pd)

    yield raises(ValueError)(__test_invalid_pade), 3
    yield raises(ValueError)(__test_invalid_pade), 6


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
        mgc = np.apply_along_axis(pysptk.mgcep, 1, windowed,
                                  filt.order, filt.alpha, gamma)
        b = np.apply_along_axis(pysptk.mgc2b, 1, mgc, filt.alpha, gamma)

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
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


def test_AllPoleDF():
    from pysptk.synthesis import AllPoleDF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(
            source, frame_len=512, hopsize=hopsize)
        lpc = np.apply_along_axis(
            pysptk.lpc, 1, windowed, filt.order)

        # make sure par has loggain
        lpc[:, 0] = np.log(lpc[:, 0])

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, lpc)
        assert np.all(np.isfinite(y))

    def __test(order):
        __test_synthesis(AllPoleDF(order))

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
        lpc = np.apply_along_axis(
            pysptk.lpc, 1, windowed, filt.order)
        par = np.apply_along_axis(pysptk.lpc2par, 1, lpc)

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
        lpc = np.apply_along_axis(
            pysptk.lpc, 1, windowed, filt.order)
        # make sure lsp has loggain
        lsp = np.apply_along_axis(pysptk.lpc2lsp, 1, lpc)
        lsp[:, 0] = np.log(lsp[:, 0])

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, lsp)
        assert np.all(np.isfinite(y))

    def __test(order):
        __test_synthesis(LSPDF(order))

    warn("TODO: tests are failing")
    # for order in [20, 25]:
    #    yield __test, order
