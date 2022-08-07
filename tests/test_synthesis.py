import numpy as np
import pysptk
import pytest
from pysptk.synthesis import Synthesizer


def __dummy_source():
    np.random.seed(98765)
    return np.random.randn(2 ** 14)


def __dummy_windowed_frames(source, frame_len=512, hopsize=80):
    np.random.seed(98765)
    n_frames = int(len(source) / hopsize) + 1
    windowed = np.random.randn(n_frames, frame_len) * pysptk.blackman(frame_len)
    return 0.5 * 32768.0 * windowed


@pytest.mark.parametrize("order", [20, 25])
@pytest.mark.parametrize("pd", [4, 5, 6, 7])
def test_LMADF(order, pd):
    from pysptk.synthesis import LMADF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(source, frame_len=512, hopsize=hopsize)
        b = pysptk.mcep(windowed, filt.order, 0.0)

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, b)
        assert np.all(np.isfinite(y))

    def __test(order, pd):
        __test_synthesis(LMADF(order, pd=pd))

    __test(order, pd)


def test_LMADF_corner_case():
    from pysptk.synthesis import LMADF

    def __test_invalid_pade(pd):
        LMADF(20, pd=pd)

    with pytest.raises(ValueError):
        __test_invalid_pade(3)
    with pytest.raises(ValueError):
        __test_invalid_pade(8)


@pytest.mark.parametrize("order", [20, 25])
@pytest.mark.parametrize("pd", [4, 5, 6, 7])
@pytest.mark.parametrize("alpha", [0.0, 0.41])
def test_MLSADF(order, pd, alpha):
    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(source, frame_len=512, hopsize=hopsize)
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

    __test(order, alpha, pd)


def test_MLSADF_corner_case():
    from pysptk.synthesis import MLSADF

    def __test_invalid_pade(pd):
        MLSADF(20, pd=pd)

    with pytest.raises(ValueError):
        __test_invalid_pade(3)
    with pytest.raises(ValueError):
        __test_invalid_pade(8)


@pytest.mark.parametrize("order", [20, 25])
@pytest.mark.parametrize("stage", [2, 5, 10])
def test_GLSADF(order, stage):
    from pysptk.synthesis import GLSADF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(source, frame_len=512, hopsize=hopsize)
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

    __test(order, stage)

    def __test_invalid_stage(stage):
        GLSADF(20, stage=stage)


def test_GLSADF_corner_case():
    from pysptk.synthesis import GLSADF

    def __test_invalid_stage(stage):
        GLSADF(20, stage=stage)

    with pytest.raises(ValueError):
        __test_invalid_stage(-1)
    with pytest.raises(ValueError):
        __test_invalid_stage(0)


@pytest.mark.parametrize("order", [20, 25])
@pytest.mark.parametrize("alpha", [0.0, 0.41])
@pytest.mark.parametrize("stage", [2, 5, 10])
def test_MGLSADF(order, alpha, stage):
    from pysptk.synthesis import MGLSADF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(source, frame_len=512, hopsize=hopsize)
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

    __test(order, alpha, stage)


def test_MGLSADF_corner_case():
    from pysptk.synthesis import MGLSADF

    def __test_invalid_stage(stage):
        MGLSADF(20, stage=stage)

    with pytest.raises(ValueError):
        __test_invalid_stage(-1)
    with pytest.raises(ValueError):
        __test_invalid_stage(0)


def test_AllZeroDF():
    pass

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(source, frame_len=512, hopsize=hopsize)
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


@pytest.mark.parametrize("order", [20, 25])
def test_AllPoleDF(order):
    from pysptk.synthesis import AllPoleDF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(source, frame_len=512, hopsize=hopsize)
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
        windowed = __dummy_windowed_frames(source, frame_len=512, hopsize=hopsize)
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

    __test(order)


@pytest.mark.parametrize("order", [20, 25])
def test_AllPoleLatticeDF(order):
    from pysptk.synthesis import AllPoleLatticeDF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(source, frame_len=512, hopsize=hopsize)
        lpc = pysptk.lpc(windowed, filt.order)
        par = pysptk.lpc2par(lpc)

        # make sure par has loggain
        par[:, 0] = np.log(par[:, 0])

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, par)
        assert np.all(np.isfinite(y))

    __test_synthesis(AllPoleLatticeDF(order))


@pytest.mark.parametrize("order", [20, 25])
def test_LSPDF(order):
    from pysptk.synthesis import LSPDF

    def __test_synthesis(filt):
        # dummy source excitation
        source = __dummy_source()

        hopsize = 80

        # dummy filter coef.
        windowed = __dummy_windowed_frames(source, frame_len=512, hopsize=hopsize)
        lpc = pysptk.lpc(windowed, filt.order)
        lsp = pysptk.lpc2lsp(lpc)
        # make sure lsp has loggain
        lsp[:, 0] = np.log(lsp[:, 0])

        # synthesis
        synthesizer = Synthesizer(filt, hopsize)
        y = synthesizer.synthesis(source, lsp)
        assert np.all(np.isfinite(y))

    __test_synthesis(LSPDF(order))
