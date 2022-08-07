import numpy as np
import pytest
from pysptk import bartlett, blackman, hamming, hanning, rectangular, trapezoid


@pytest.mark.parametrize(
    "f", [blackman, hanning, hamming, bartlett, trapezoid, rectangular]
)
@pytest.mark.parametrize("n", [16, 128, 256, 1024, 2048, 4096])
def test_windows(f, n):
    def __test(f, N, normalize):
        w = f(N, normalize)
        assert np.all(np.isfinite(w))

    __test(f, n, 1)


def test_windows_corner_case():
    def __test(f, N, normalize):
        f(N, normalize)

    # unsupported normalize flags
    for f in [blackman, hanning, hamming, bartlett, trapezoid, rectangular]:
        with pytest.raises(ValueError):
            __test(f, 256, -1)
        with pytest.raises(ValueError):
            __test(f, 256, 3)
