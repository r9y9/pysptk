# coding: utf-8

import numpy as np
import pysptk
from nose.tools import raises
from pysptk.util import mcepalpha


def test_assert_gamma():
    def __test(gamma):
        pysptk.util.assert_gamma(gamma)

    for gamma in [-2.0, 0.1]:
        yield raises(ValueError)(__test), gamma


def test_assert_pade():
    def __test(pade):
        pysptk.util.assert_pade(pade)

    for pade in [3, 6]:
        yield raises(ValueError)(__test), pade


def test_assert_fftlen():
    def __test(fftlen):
        pysptk.util.assert_fftlen(fftlen)

    for fftlen in [255, 257]:
        yield raises(ValueError)(__test), fftlen


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


def test_example_audio_file():
    from os.path import exists
    path = pysptk.util.example_audio_file()
    assert exists(path)


def test_mcepalpha():
    assert np.isclose(mcepalpha(8000), 0.312)
    assert np.isclose(mcepalpha(11025), 0.357)
    assert np.isclose(mcepalpha(16000), 0.41)
    assert np.isclose(mcepalpha(22050), 0.455)
    assert np.isclose(mcepalpha(44100),  0.544)
    assert np.isclose(mcepalpha(48000), 0.554)
