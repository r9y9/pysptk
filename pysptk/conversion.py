# coding: utf-8

"""
Other conversions
-----------------
.. autosummary::
    :toctree: generated/

    mgc2b

"""

from __future__ import division, print_function, absolute_import

import numpy as np

from pysptk.sptk import mc2b, gnorm


def mgc2b(mgc, alpha=0.35, gamma=0.0):
    """Mel-generalized cepstrum to MGLSA filter coefficients

    Parameters
    ----------
    mgc : array, shape
        Mel-generalized cepstrum

    alpha : float
        All-pass constant. Default is 0.35.

    gamma : float
        Parameter of generalized log function. Default is 0.0.

    Returns
    -------
    b : array, shape(same as ``mgc``)
        MGLSA filter coefficients

    See Also
    --------
    pysptk.sptk.mlsadf
    pysptk.sptk.mglsadf
    pysptk.sptk.mc2b
    pysptk.sptk.b2mc
    pysptk.sptk.mcep
    pysptk.sptk.mgcep
    pysptk.sptk.mgcep

    """

    b = mc2b(mgc, alpha)
    if gamma == 0:
        return b

    b = gnorm(b, gamma)

    b[0] = np.log(b[0])
    b[1:] *= gamma

    return b
