# coding: utf-8

"""
A python wrapper for `Speech Signal Processing Toolkit (SPTK)
<http://sp-tk.sourceforge.net>`_.

https://github.com/r9y9/pysptk

Full documentation
------------------

A full documentation of SPTK is available at http://sp-tk.sourceforge.net.

The wrapper is based on a modified version of SPTK (`r9y9/SPTK
<https://github.com/r9y9/SPTK>`_)
"""

from __future__ import division, print_function, absolute_import

import pkg_resources

__version__ = pkg_resources.get_distribution('pysptk').version

from .sptk import *  # pylint: disable=wildcard-import

from . import synthesis
from .conversion import mgc2b
from . import util
