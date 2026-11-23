Installation guide
==================

Install pysptk from PyPI.

.. code-block:: bash

   python -m pip install pysptk

Install the development version directly from GitHub.

.. code-block:: bash

   python -m pip install git+https://github.com/r9y9/pysptk

To build from a local checkout, clone the repository with its SPTK submodule.

.. code-block:: bash

   git clone --recurse-submodules https://github.com/r9y9/pysptk.git
   cd pysptk
   python -m pip install .

Source builds require a C compiler. Pip installs Cython and the NumPy headers
in an isolated build environment. On Windows, use the Microsoft C++ Build Tools.
