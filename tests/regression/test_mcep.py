# coding: utf-8

import numpy as np
import pysptk
from os.path import join, dirname

DATA_DIR = join(dirname(__file__), "..", "data")


def test_mcep_from_windowed_frames():
    # frame -l 512 -p 80 < test16k.float | window -l 512 | mcep -q 0 -l 512 -a \
    # 0.41 -m 24 | dmp +f | awk '{print $2}' > test16k_mcep.txt
    mc = np.loadtxt(
        join(DATA_DIR, "test16k_mcep.txt")).reshape(759, 25).astype(np.float64)

    # frame -l 512 -p 80 < test16k.float | window -l 512 | dmp +f | awk \
    # '{print $2}' > test16k_windowed.txt
    frames = np.loadtxt(
        join(DATA_DIR, "test16k_windowed.txt")).reshape(759, 512).astype(np.float64)
    mc_hat = np.apply_along_axis(
        pysptk.mcep, 1, frames, order=24, alpha=0.41)

    assert mc.shape == mc_hat.shape
    assert np.allclose(mc, mc_hat, atol=5e-4)  # TODO: should be smaller?


def test_mcep_from_H():
    """Test mel-cepstrum computation from power spectrum (|H(w)|^2)
    """
    #  frame -l 512 -p 80 < test16k.float | window -l 512 | mcep -q 0 -l 512 \
    # -a 0.41 -m 24 | mgc2sp -m 24 -a 0.41 -g 0 -l 512 -o 3 | mcep -q 4 -l 512 \
    # -a 0.41 -m 24 | dmp +f  | awk '{print $2}' > test16k_mcep_from_H.txt
    mc = np.loadtxt(
        join(DATA_DIR, "test16k_mcep_from_H.txt")).reshape(759, 25).astype(np.float64)

    # frame -l 512 -p 80 < test16k.float | window -l 512 | mcep -q 0 -l 512 -a \
    # 0.41 -m 24 | mgc2sp -m 24 -a 0.41 -g 0 -l 512 -o 3 | dmp +f | awk \
    # '{print $2}' > test16k_H.txt
    H = np.loadtxt(join(DATA_DIR, "test16k_H.txt")).reshape(
        759, 257).astype(np.float64)
    mc_hat = np.apply_along_axis(
        pysptk.mcep, 1, H, order=24, alpha=0.41, itype=4)

    assert mc.shape == mc_hat.shape
    assert np.allclose(mc, mc_hat, atol=1e-6)


def test_mgc2sp():
    """mgc2sp

    ref: https://github.com/r9y9/pysptk/issues/57
    """
    # frame -l 512 -p 80 < test16k.float | window -l 512| mcep -q 0 -l 512 \
    # -a 0.41 -m 24 | mgc2sp -a 0.41 -m 24 -l 512 -o 3 > test16k_57.sp
    # output type 3 means we get power spectrum |H(w)|^2
    sp = np.fromfile(
        join(DATA_DIR, "test16k_57.sp"), dtype=np.float32).reshape(759, 257).astype(np.float64)

    # frame -l 512 -p 80 < test16k.float | window -l 512 \
    # | mcep -q 0 -l 512 -a 0.41 -m 24 > test16k_57.mgc
    mgc = np.fromfile(join(DATA_DIR, "test16k_57.mgc"), dtype=np.float32).reshape(
        759, 25).astype(np.float64)

    # mgc2sp does conversion: c(k) -> log H(w)
    # so convert it to |H(w)|^2 to get power spectrum
    sp_hat = np.exp(pysptk.mgc2sp(mgc, 0.41, 0, 512).real*2)

    assert np.allclose(sp, sp_hat)
