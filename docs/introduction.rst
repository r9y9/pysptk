Installation guide
==================

The latest release is availabe on pypi. You can install it by:

.. code::

    pip install pysptk

Note that you have to install ``numpy`` to build C-extensions.

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


For Windows users
^^^^^^^^^^^^^^^^^

There are some binary wheels available on pypi, so you can install ``pysptk`` via pip **without cython and C compilier** if there exists a binary wheel that matches your environment (depends on bits of system and python version). For now, wheels are avilable for:

* Python 2.7 on 32 bit system
* Python 2.7 on 64 bit system
* Python 3.4 on 32 bit system

If there is no binary wheel available for your environment, you can build ``pysptk`` from the source distribution, which is also available on pypi. Note that in order to compile ``pysptk`` from source in Windows, it is highly recommended to use `Anaconda
<https://github.com/r9y9/SPTK>`_ , since installation of numpy, cython and other scientific packages is really easy. In fact, continuous integration in Windows on AppVeyor uses Anacona to build and test ``pysptk``.  See `pysptk/appveyor.yml <https://github.com/r9y9/pysptk/blob/master/appveyor.yml>`_ for the exact build steps.
