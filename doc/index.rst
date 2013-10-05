=========================================================
:mod:`vsm` --- Vector Space Modeling of Textual Semantics
=========================================================

:Release: |version|
:Date: |today|

The :mod:`vsm` module provides tools and a workflow for producing
semantic models of textual corpora and analyzing and visualizing these
models.

The :mod:`vsm` module has been conceived within the :ref:`SciPy`
ecosystem. In a typical workflow, a collection of texts is first
transformed into a :class:`Corpus` object, whose underlying data
structures are :ref:`NumPy` numerical :class:`arrays`. The user may
then feed a :class:`Corpus` object to one of the :mod:`model` classes,
which contain the algorithms, implemented in :ref:`NumPy`,
:ref:`SciPy` and :ref:`IPython.parallel`, for training models such as
TF, TFIDF, LSA, BEAGLE, or LDA. Finally, the user may examine the
results with a :mod:`viewer` class specialized to a particular model
type. A :mod:`viewer` class contains a variety of methods for analysis
and graphical display and achieves its full functionality within an
:ref:`IPython` notebook session extended with :ref:`matplotlib` and
:ref:`scikit-learn`.

This documentation site is currently in a pre-alpha state.

.. toctree::
   :maxdepth: 2

   workflow

.. Indices and tables
   ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


.. Acknowledgements
.. ================
