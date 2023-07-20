Change log
==========

v0.2.1 <2023-07-20>
--------------------

- `#96`_: Restrict Cython version to <3.0.0

v0.2.0 <2022-11-06>
--------------------

Newer numpy  (>v1.20.0) is required to avoid ABI compatibility issues. Please check the updated installation guide.

- `#92`_: Support for Python 3.10, rewrite tests with pytest

v0.1.21 <2022-06-05>
--------------------

Fixes windows build issues caused by intrinsic conflicts


v0.1.20 <2022-01-04>
--------------------

Fixes for ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject

v0.1.19 <2022-01-04>
--------------------

- `#87`_: Add pysen and format all the codes. Appveyor is now disabled. Windows CI is done by Github Actions.
- `#83`_: pitch: support automatic type converseion for swipe and rapt (API change)

v0.1.18 <2020-03-02>
--------------------

- `#75`_: Enable py38 test on travis
- `#72`_: Fix itype!=0 for mgcep, gcep and related functions


v0.1.17 <2019-05-31>
--------------------

- Fixed python 3.7 installation problem

v0.1.16 <2019-02-17>
--------------------

- Fixed cython-generated files in the release tarball with latest numpy

v0.1.15 <2019-01-08>
--------------------

- `#65`_, #67`_: Should avoid build-time numpy dependency for newer releases

v0.1.14 <2018-11-25>
--------------------

- `#62`_, `#64`_: **IMPORTANT**: Add ``use_scipy`` option to ``levdur`` and ``lpc``, and set it to True by default to improve numerical stability. The results should be same regardless to the option, but this would be technically a breaking change.
- `#64`_: Add ``acorr``.

v0.1.13 <2018-11-19>
--------------------

- Add All zero synthesis filter.
- Add ``levdur``.
- Add tranposed synthesis filters (``mlsadft``, ``poledft``, ``mglsadft``, ``glsadft``)
- Add missing high level synthesis filter class ``GLSADF``.

v0.1.12 <2018-10-27>
--------------------

- `#63`_: Fix lpc2lsp bug, add lsp2lpc function. Add regression tests for those.

v0.1.11 <2018-02-05>
--------------------

- `#55`_: Add numpy implementation of cdist

v0.1.10 <2018-01-02>
--------------------

- `#54`_: Changes from SPTK v3.11 release. 6 and 7 pade approximatino in lmadf and mlsadf is now supported,

v0.1.9 <2018-01-01>
-------------------

- BUG fix: example_audio_data is now included in the release tar.gz


v0.1.8 <2017-12-25>
-------------------

-  c2acr: Fix segfaults for small fftsize

v0.1.7 <2017-06-28>
-------------------

-  Extend vec2vec functions to mat2mat
   `#49 <https://github.com/r9y9/pysptk/issues/49>`__
-  Support automatic type conversions
   `#48 <https://github.com/r9y9/pysptk/issues/48>`__

v0.1.6 <2017-05-18>
-------------------

-  Add ``mcepalpha``. `#43 <https://github.com/r9y9/pysptk/issues/43>`__
-  Add ``mc2e``. `#42 <https://github.com/r9y9/pysptk/pull/42>`__
-  Add ``sp2mc`` and ``mc2sp``.
   `#41 <https://github.com/r9y9/pysptk/pull/41>`__

v0.1.5 <2017-04-22>
-------------------

-  Fix mcep eps check and input length
   `#39 <https://github.com/r9y9/pysptk/pull/39>`__

v0.1.4 <2015-11-23>
-------------------

-  Add developer documentation
   (`#34 <https://github.com/r9y9/pysptk/issues/34>`__)
-  Separate cython implementation and interface
   (`#35 <https://github.com/r9y9/pysptk/pull/35>`__)
-  Add RAPT (`#32 <https://github.com/r9y9/pysptk/pull/32>`__)
-  Add excite function
   (`#31 <https://github.com/r9y9/pysptk/pull/31>`__)
   `@jfsantos <https://github.com/jfsantos>`__
-  Fix inconsistent docs about normalization flag for window functions
-  Fix test failure in c2dps / ndps2c
   (`#29 <https://github.com/r9y9/pysptk/issues/29>`__)

v0.1.3 <2015-10-02>
-------------------

-  Building binary wheels for Windows using Appveyor
   (`#28 <https://github.com/r9y9/pysptk/pull/28>`__)
-  Add Installation guide on windows
   (`#25 <https://github.com/r9y9/pysptk/issues/25>`__)
-  Start Windows continuous integration on AppVeyor
   (`#24 <https://github.com/r9y9/pysptk/pull/24>`__). As part of the
   issue, binary dependency was updated so that SPTK library can be
   compiled on linux, osx and Windows as well.
-  Remove unnecesarry array initialization
   (`#23 <https://github.com/r9y9/pysptk/pull/23>`__)

v0.1.2 <2015-09-12>
-------------------

-  Add ``pysptk.synthesis`` package that provides high level interfaces
   for speech waveform synthesis
   (`#14 <https://github.com/r9y9/pysptk/pull/14>`__)
-  Add cross-link to the docs
-  Add ``pysptk.conversion.mgc2b``
-  Add speech analysis and re-synthesis demonstration notebook
   (`#13 <https://github.com/r9y9/pysptk/issues/13>`__)
-  Add ``pysptk.util.example_audio_file``
-  Add ``fftcep`` (`#18 <https://github.com/r9y9/pysptk/issues/18>`__)
-  Add ``mfcc`` (`#21 <https://github.com/r9y9/pysptk/pull/21>`__)
-  Cython is now only required to build development versioni of pysptk.
   (`#8 <https://github.com/r9y9/pysptk/issues/8>`__)

v0.1.1 <2015-09-05>
-------------------

-  Include \*.c to pypi distribution

v0.1.0 <2015-09-05>
-------------------

-  Initial release

.. _#54: https://github.com/r9y9/pysptk/pull/54
.. _#55: https://github.com/r9y9/pysptk/issues/55
.. _#63: https://github.com/r9y9/pysptk/pull/63
.. _#62: https://github.com/r9y9/pysptk/issues/62
.. _#64: https://github.com/r9y9/pysptk/pull/64
.. _#65: https://github.com/r9y9/pysptk/issues/65
.. _#67: https://github.com/r9y9/pysptk/pull/67
.. _#72: https://github.com/r9y9/pysptk/issues/72
.. _#75: https://github.com/r9y9/pysptk/pull/75
.. _#83: https://github.com/r9y9/pysptk/pull/83
.. _#87: https://github.com/r9y9/pysptk/pull/87
.. _#92: https://github.com/r9y9/pysptk/pull/92
.. _#96: https://github.com/r9y9/pysptk/pull/96
