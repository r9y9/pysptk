# coding: utf-8

import numpy as np
from nose.tools import raises

from pysptk import blackman, hanning, hamming
from pysptk import bartlett, trapezoid, rectangular


def test_windows():
    def __test(f, N, normalize):
        w = f(N, normalize)
        assert np.all(np.isfinite(w))

    for f in [blackman, hanning, hamming, bartlett, trapezoid, rectangular]:
        for n in [16, 128, 256, 1024, 2048, 4096]:
            yield __test, f, n, 1

    # unsupported normalize flags
    for f in [blackman, hanning, hamming, bartlett, trapezoid, rectangular]:
        yield raises(ValueError)(__test), f, 256, -1
        yield raises(ValueError)(__test), f, 256, 3
