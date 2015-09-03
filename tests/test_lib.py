# coding: utf-8

import numpy as np
import six
import pysptk


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
