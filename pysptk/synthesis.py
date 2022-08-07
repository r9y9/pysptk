"""
High-level interface for waveform synthesis
===========================================

Module ``pysptk.synthesis`` provides high-leve interface that wraps low-level
SPTK waveform synthesis functions (e.g. ``mlsadf``),

Synthesis filter interface
--------------------------

.. autoclass:: SynthesisFilter
    :members:

Synthesizer
-----------
.. autoclass::  Synthesizer
    :members:

SynthesisFilters
----------------

LMADF
^^^^^
.. autoclass::  LMADF
    :members:

MLSADF
^^^^^^
.. autoclass::  MLSADF
    :members:

GLSADF
^^^^^^
.. autoclass::  GLSADF
    :members:

MGLSADF
^^^^^^^
.. autoclass::  MGLSADF
    :members:

AllZeroDF
^^^^^^^^^
.. autoclass::  AllZeroDF
    :members:

AllPoleDF
^^^^^^^^^
.. autoclass::  AllPoleDF
    :members:

AllPoleLatticeDF
^^^^^^^^^^^^^^^^
.. autoclass::  AllPoleLatticeDF
    :members:

"""

from abc import abstractmethod

import numpy as np
import pysptk
from pysptk.util import assert_pade, assert_stage


class SynthesisFilter(object):
    """Synthesis filter interface

    All synthesis filters must implement this interface.
    """

    @abstractmethod
    def filt(self, x, coef):
        """Filter one sample

        Parameters
        ----------
        x : float
            A input sample

        coef : array
            Filter coefficients

        Returns
        -------
        y : float
            A filtered sample

        """

    def filtt(self, x, coef):
        """Transpose filter

        Can be optional.

        Parameters
        ----------
        x : float
            A input sample

        coef : array
            Filter coefficients

        Returns
        -------
        y : float
            A filtered sample

        """
        raise NotImplementedError()


class Synthesizer(object):
    """Speech waveform synthesizer


    Attributes
    ----------
    filt : SynthesisFilter
        A speech synthesis filter

    hopsize : int
        Hop size

    transpose : bool
        Transpose filter or not. Default is False.

    """

    def __init__(self, filt, hopsize, transpose=False):
        """Initialization

        Raises
        ------
        TypeError
            if ``filt`` is not a instance of ``SynthesisFilter``

        """
        if not isinstance(filt, SynthesisFilter):
            raise TypeError("filt must be an instance of SynthesisFilter")

        self.filt = filt
        self.hopsize = hopsize
        self.transpose = transpose

    def synthesis_one_frame(self, source, prev_b, curr_b):
        """Synthesize one frame waveform

        Parameters
        ----------
        source : array
            Source excitation

        prev_b : array
            Filter coefficients of previous frame

        curr_b : array
            Filter coefficients of current frame

        Returns
        -------
        y : array
            Synthesized waveform

        """
        assert len(prev_b) == len(curr_b)
        slope = (curr_b - prev_b) / len(source)
        interpolated_coef = prev_b.copy()

        y = np.empty_like(source)

        # filter function
        filt = self.filt.filtt if self.transpose else self.filt.filt

        for i in range(len(source)):
            scaled_source = source[i] * np.exp(interpolated_coef[0])
            y[i] = filt(scaled_source, interpolated_coef)
            interpolated_coef += slope

        return y

    def synthesis(self, source, b):
        """Synthesize a waveform given a source excitation and sequence of
        filter coefficients (e.g. cepstrum).

        Parameters
        ----------
        source : array
            Source excitation

        b : array
            Filter coefficients

        Returns
        -------
        y : array, shape (same as ``source``)
            Synthesized waveform

        """

        y = np.zeros_like(source)

        b_prev = b[0, :]
        b_curr = b_prev
        for i in range(b.shape[0]):
            if i > 0:
                b_prev = b[i - 1, :]
            b_curr = b[i, :]

            s, e = i * self.hopsize, (i + 1) * self.hopsize
            if e > len(source):
                break

            y[s:e] = self.synthesis_one_frame(source[s:e], b_prev, b_curr)

        return y


class LMADF(SynthesisFilter):
    """LMA digital filter that wraps ``lmadf``

    Attributes
    ----------
    pd : int
        Order of pade approximation. Default is 4.

    delay : array
        Delay

    """

    def __init__(self, order=25, pd=4):
        """Initialization

        Raises
        ======
        ValueError
            if invalid order of pade approximation is specified

        """
        self.order = order

        assert_pade(pd)

        self.pd = pd
        self.delay = pysptk.lmadf_delay(order, pd)

    def filt(self, x, coef):
        """Filter one sample using ``lmadf``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            LMA filter coefficients (i.e. Cepstrum)

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.lmadf

        """

        return pysptk.lmadf(x, coef, self.pd, self.delay)


class MLSADF(SynthesisFilter):
    """MLSA digital filter that wraps ``mlsadf``

    Attributes
    ----------
    alpha : float
        All-pass constant

    pd : int
        Order of pade approximation. Default is 4.

    delay : array
        Delay

    """

    def __init__(self, order=25, alpha=0.35, pd=4):
        """Initialization

        Raises
        ======
        ValueError
            if invalid order of pade approximation is specified

        """

        self.order = order

        assert_pade(pd)

        self.alpha = alpha
        self.pd = pd
        self.delay = pysptk.mlsadf_delay(order, pd)

    def filt(self, x, coef):
        """Filter one sample using ``mlsadf``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            MLSA filter coefficients

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.mlsadf
        pysptk.sptk.mc2b

        """
        return pysptk.mlsadf(x, coef, self.alpha, self.pd, self.delay)

    def filtt(self, x, coef):
        """Transpose filter using ``mlsadft``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            MLSA filter coefficients

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.mlsadft
        """
        return pysptk.mlsadft(x, coef, self.alpha, self.pd, self.delay)


class GLSADF(SynthesisFilter):
    """GLSA digital filter that wraps ``glsadf``

    Attributes
    ----------
    stage : int
        -1/gamma

    delay : array
        Delay

    """

    def __init__(self, order=25, stage=1):
        """Initialization

        Raises
        ------
        ValueError
            if invalid number of stage is specified

        """
        self.order = order

        assert_stage(stage)

        self.stage = stage
        self.delay = pysptk.glsadf_delay(order, stage)

    def filt(self, x, coef):
        """Filter one sample using ``glsadf``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            GLSA filter coefficients

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.glsadf
        """
        return pysptk.glsadf(x, coef, self.stage, self.delay)

    def filtt(self, x, coef):
        """Filter one sample using ``glsadft``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            GLSA filter coefficients

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.glsadft
        """
        return pysptk.glsadft(x, coef, self.stage, self.delay)


class MGLSADF(SynthesisFilter):
    """MGLSA digital filter that wraps ``mglsadf``

    Attributes
    ----------
    alpha : float
        All-pass constant

    stage : int
        -1/gamma

    delay : array
        Delay

    """

    def __init__(self, order=25, alpha=0.35, stage=1):
        """Initialization

        Raises
        ------
        ValueError
            if invalid number of stage is specified

        """
        self.order = order

        assert_stage(stage)

        self.alpha = alpha
        self.stage = stage
        self.delay = pysptk.mglsadf_delay(order, stage)

    def filt(self, x, coef):
        """Filter one sample using ``mglsadf``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            MGLSA filter coefficients

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.mglsadf
        """
        return pysptk.mglsadf(x, coef, self.alpha, self.stage, self.delay)

    def filtt(self, x, coef):
        """Filter one sample using ``mglsadft``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            MGLSA filter coefficients

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.mglsadft
        """
        return pysptk.mglsadft(x, coef, self.alpha, self.stage, self.delay)


class AllZeroDF(SynthesisFilter):
    """All-zero digital filter that wraps ``zerodf``

    Attributes
    ----------
    delay : array
        Delay

    """

    def __init__(self, order=25):
        """Initialization"""

        self.order = order
        self.delay = pysptk.zerodf_delay(order)

    def filt(self, x, coef):
        """Filter one sample using using ``zerodf``

                Parameters
                ----------
                x : float
                    A input sample

                coef: array
                    FIR parameters
        _
                Returns
                -------
                y : float
                    A filtered sample

                See Also
                --------
                pysptk.sptk.zerodf
        """
        return pysptk.zerodf(x, coef, self.delay)

    def filtt(self, x, coef):
        """Filter one sample using using ``zerodft``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            FIR parameters

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.zerodft
        """
        return pysptk.zerodf(x, coef, self.delay)


class AllPoleDF(SynthesisFilter):
    """All-pole digital filter that wraps ``poledf``

    Attributes
    ----------
    delay : array
        Delay

    """

    def __init__(self, order=25):
        """Initialization"""

        self.order = order
        self.delay = pysptk.poledf_delay(order)

    def filt(self, x, coef):
        """Filter one sample using using ``poledf``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            LPC (with loggain)

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.poledf
        """
        return pysptk.poledf(x, coef, self.delay)

    def filtt(self, x, coef):
        """Filter one sample using using ``poledft``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            LPC (with loggain)

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.poledft
        """
        return pysptk.poledf(x, coef, self.delay)


class AllPoleLatticeDF(SynthesisFilter):
    """All-pole lttice digital filter that wraps ``ltcdf``

    Attributes
    ----------
    delay : array
        Delay

    """

    def __init__(self, order=25):
        """Initialization"""

        self.order = order
        self.delay = pysptk.ltcdf_delay(order)

    def filt(self, x, coef):
        """Filter one sample using using ``ltcdf``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            PARCOR coefficients (with loggain)

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.ltcdf

        """

        return pysptk.ltcdf(x, coef, self.delay)


class LSPDF(SynthesisFilter):
    """All-pole digital filter that wraps ``lspdf``

    Attributes
    ----------
    delay : array
        Delay

    """

    def __init__(self, order=25):
        """Initialization"""

        self.order = order
        self.delay = pysptk.lspdf_delay(order)

    def filt(self, x, coef):
        """Filter one sample using using ``lspdf``

        Parameters
        ----------
        x : float
            A input sample

        coef: array
            LSP (with loggain)

        Returns
        -------
        y : float
            A filtered sample

        See Also
        --------
        pysptk.sptk.lspdf

        """

        return pysptk.lspdf(x, coef, self.delay)
