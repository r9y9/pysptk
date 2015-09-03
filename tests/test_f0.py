# coding: utf-8

import numpy as np
import pysptk
from nose.tools import raises


def test_swipe():

    def __test(x, fs, hopsize, otype):
        f0 = pysptk.swipe(x, fs, hopsize, otype=otype)
        assert np.all(np.isfinite(f0))

    np.random.seed(98765)
    fs = 16000
    x = np.random.rand(16000)

    for hopsize in [40, 80, 160, 320]:
        for otype in [0, 1, 2]:
            yield __test, x, fs, hopsize, otype

    # unsupported otype
    yield raises(ValueError)(__test), x, fs, 80, -1
    yield raises(ValueError)(__test), x, fs, 80, 3
