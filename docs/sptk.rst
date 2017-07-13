Core SPTK API
=============

All functionality in ``pysptk.sptk`` (the core API) is directly accesible from the top-level ``pysptk.*`` namespace.


For convenience, vector-to-vector functions (``pysptk.mcep``, ``pysptk.mc2b``, etc) that takes an input vector as the first argment, can also accept matrix. As for matrix inputs, vector-to-vector functions are applied along with the last axis internally; e.g.

.. code::

   mc = pysptk.mcep(frames) # frames.shape == (num_frames, frame_len)

is equivalent to:

.. code::

   mc = np.apply_along_axis(pysptk.mcep, -1, frames)


.. warning:: The core APIs in ``pysptk.sptk`` package are based on the SPTK's internal APIs (e.g. code in ``_mgc2sp.c``), so the functionalities are not exactly same as SPTK's CLI. If you find any inconsistency that should be addressed, please file an issue.

.. note:: Almost all of pysptk functions assume that the input array is **C-contiguous** and has ``float64`` element type. For vector-to-vector functions, the input array is automatically converted to ``float64``-typed one, the function is executed on it, and then the output array is converted to have the same type with the input you provided.

.. note::

.. automodule:: pysptk.sptk
