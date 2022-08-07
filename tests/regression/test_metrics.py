from os.path import dirname, join

import numpy as np
import pysptk
import pytest

DATA_DIR = join(dirname(__file__), "..", "data")


def test_cdist_invalid():
    c1 = np.random.rand(10, 26)
    c2 = np.random.rand(10, 26)

    def __test_invalid(otype):
        pysptk.cdist(c1, c2, otype=otype)

    with pytest.raises(ValueError):
        __test_invalid(-1)
    with pytest.raises(ValueError):
        __test_invalid(3)


def test_cdist():
    order = 24
    # frame -l 512 -p 80 < test16k.float | window -l 512 \
    # | mcep -q 0 -l 512 -a 0.41 -m 24 | freqt -a 0.41 -m 24 -A 0 -M 24 > test16k.mcep.cep
    c1 = np.fromfile(join(DATA_DIR, "test16k.mcep.cep"), dtype=np.float32).reshape(
        -1, order + 1
    )
    # frame -l 512 -p 80 < test16k.float | window -l 512 | fftcep -m 24 -l 512 > test16k.cep
    c2 = np.fromfile(join(DATA_DIR, "test16k.cep"), dtype=np.float32).reshape(
        -1, order + 1
    )

    assert c1.shape == c2.shape
    # cdist test16k.cep -m 24 -o 0 < test16k.mcep.cep | dmp +f
    assert np.allclose(pysptk.cdist(c1, c2), 1.89798)
    assert np.allclose(pysptk.cdist(c1, c2, otype=0), 1.89798)
    # cdist test16k.cep -m 24 -o 1 < test16k.mcep.cep | dmp +f
    assert np.allclose(pysptk.cdist(c1, c2, otype=1), 0.10249)
    # cdist test16k.cep -m 24 -o 2 < test16k.mcep.cep | dmp +f
    assert np.allclose(pysptk.cdist(c1, c2, otype=2), 0.309023)

    # cdist test16k.cep -m 24 -o 0 -f < test16k.mcep.cep > test16k.cdist
    d = np.fromfile(join(DATA_DIR, "test16k.cdist"), dtype=np.float32)
    d_hat = pysptk.cdist(c1, c2, otype=0, frame=True)
    assert np.allclose(d, d_hat)
