Installation guide
==================

The latest release is availabe on pypi. Assuming you have already ``numpy`` installed, you can install pysptk by:

.. code::

    pip install pysptk

If yout want the latest development version, run:

.. code::

   pip install git+https://github.com/r9y9/pysptk

or:

.. code::

   git clone https://github.com/r9y9/pysptk
   cd pysptk
   python setup.py develop # or install

This should resolve the package dependencies and install ``pysptk`` property.


.. note::

   If you use the development version, you need to have ``cython`` (and C compiler) installed to compile cython module(s).

Workaround for ``ValueError: numpy.ndarray size changed, may indicate binary incompatibility``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This type of errors comes from the Numpys' ABI breaking changes. If you see ``ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject`` or similar, please make sure to install numpy first, and then install pyopenjtalk by:

.. code::

   pip install pysptk --no-build-isolation

or:

.. code::

   pip install git+https://github.com/r9y9/pysptk --no-build-isolation

The option ``--no-build-isolation`` tells pip not to create a build environment, so the pre-installed numpy is used to build the packge. Hense there should be no Numpy's ABI issues.

For Windows users
^^^^^^^^^^^^^^^^^

There are some binary wheels available on pypi, so you can install ``pysptk`` via pip **without cython and C compilier** if there exists a binary wheel that matches your environment (depends on bits of system and python version). For now, wheels are available for:

* Python 2.7 on 32 bit system
* Python 2.7 on 64 bit system
* Python 3.4 on 32 bit system

If there is no binary wheel available for your environment, you can build ``pysptk`` from the source distribution, which is also available on pypi. Note that in order to compile ``pysptk`` from source in Windows, it is highly recommended to use `Anaconda
<https://github.com/r9y9/SPTK>`_ , since installation of numpy, cython and other scientific packages is really easy. In fact, continuous integration in Windows on AppVeyor uses Anacona to build and test ``pysptk``.  See `pysptk/appveyor.yml <https://github.com/r9y9/pysptk/blob/master/appveyor.yml>`_ for the exact build steps.
