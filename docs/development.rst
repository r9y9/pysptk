Developer Documentation
=======================

Design principle
----------------

pysptk is a thin python wrapper of SPTK. It is designed to be API consistent
with the original SPTK as possible, but give better interface. There are a few
design principles to wrap C interface:

1. Avoid really short names for variables (e.g. a, b, c, aa, bb, dd)

    Variable names should be informative. If the C functions have such short
    names, use self-descriptive names instead for python interfaces, unless
    they have clear meanings in their context.

2. Avoid too many function arguments

    Less is better. If the C functions have too many function arguments, use
    keyword arguments with proper default values for optional ones in python.

3. Handle errors in python

    Since C functions might `exit` (unfortunately) inside their functions for
    unexpected inputs, it should be check if the inputs are supported or not
    in python.

To wrap C interface, Cython is totally used.


How to build pysptk
-------------------

You have to install ``numpy`` and ``cython`` first, and then:

::

    git clone https://github.com/r9y9/pysptk
    cd pysptk
    git submodule update --init
    python setup.py develop

should work.

.. note::

    Dependency to the SPTK is added as a submodule. You have to checkout the
    supported SPTK as ``git sudmobule update --init`` before running setup.py.


How to build docs
-----------------

pysptk docs are managed by the python sphinx. Docs-related dependencies can be
resolved by:

.. code::

    pip install .[docs]

at the top of pysptk directory.

To build docs, go to the `docs` directory and then:

.. code::

    make html

You will see the generated docs in `_build` directory as follows (might
different depends on sphinx version):

::

    % tree _build/ -d
    _build/
    ├── doctrees
    │   └── generated
    ├── html
    │   ├── _images
    │   ├── _modules
    │   │   └── pysptk
    │   ├── _sources
    │   │   └── generated
    │   ├── _static
    │   │   ├── css
    │   │   ├── fonts
    │   │   └── js
    │   └── generated
    └── plot_directive
        └── generated


See `_build/html/index.html` for the top page of the generated docs.


How to add a new function
-------------------------

There are a lot of functions unexposed from SPTK. To add a new function to pysptk,
there are a few typical steps:

    1. Add function signature to ``_sptk.pxd``
    2. Add cython implementation to ``_sptk.pyx``
    3. Add python interface (with docstrings) to ``sptk.py`` (or some proper module)

As you can see in setup.py, ``_sptk.pyx`` and SPTK sources are compiled into a
single extension module.

.. note::

    You might wonder why cython implementation and python interface should be
    separated because cython module can be directly accessed by python. The
    reasons are 1) to avoid rebuilding cython module when docs strings are
    changed in the source 2) to make doc looks great, since sphinx seems
    unable to collect function argments correctly from cython module for now.
    Relevant issue: `pysptk/#33`_

.. _pysptk/#33: https://github.com/r9y9/pysptk/issues/33


An example
~~~~~~~~~~~~~~~

In _sptk.pyd:

.. code::

    cdef extern from "SPTK.h":
        double _agexp "agexp"(double r, double x, double y)

In _sptk.pyx:

.. code::

    def agexp(r, x, y):
        return _agexp(r, x, y)

In sptk.pyx:

.. code::

    def agexp(r, x, y):
        """Magnitude squared generalized exponential function

        Parameters
        ----------
        r : float
            Gamma
        x : float
            Real part
        y : float
            Imaginary part

        Returns
        -------
        Value

        """
        return _sptk.agexp(r, x, y)
