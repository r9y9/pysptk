# coding: utf-8

"""
A python wrapper for `Speech Signal Processing Toolkit (SPTK)
<http://sp-tk.sourceforge.net>`_.

https://github.com/r9y9/pysptk

The wrapper is based on a modified version of SPTK (`r9y9/SPTK`_)

.. _r9y9/SPTK: https://github.com/r9y9/SPTK

Full documentation
------------------

A full documentation of SPTK is available at http://sp-tk.sourceforge.net.
If you are not familiar with SPTK, I recommend you to take a look at the doc
first before using ``pysptk``.

Demonstration notebooks
-----------------------

* `Introduction notebook`_: a brief introduction to pysptk
* `Speech analysis and re-synthesis resynthesis notebook`_: a demonstration \
notebook for speech analysis and re-synthesis. Synthesized audio examples\
(English) are available on the notebook.

.. _Introduction notebook: \
http://nbviewer.ipython.org/github/r9y9/pysptk/blob/master/examples/\
pysptk%20introduction.ipynb
.. _Speech analysis and re-synthesis resynthesis notebook:\
 http://nbviewer.ipython.org/github/r9y9/pysptk/blob/master/examples/\
Speech%20analysis%20and%20re-synthesis.ipynb

"""

from . import synthesis, util  # noqa
from .conversion import mc2e, mc2sp, mgc2b, sp2mc  # noqa
from .sptk import *  # noqa
from .version import __version__  # noqa
