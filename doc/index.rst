.. vsm documentation master file, created by
   sphinx-quickstart on Thu Jul  4 13:59:01 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=========================================================
:mod:`vsm` --- Vector Space Modeling of Textual Semantics
=========================================================

:Release: |version|
:Date: |today|

The :mod:`vsm` module provides tools and a work flow for producing
semantic models of textual corpora and analyzing and visualizing these
models.

The :mod:`vsm` module has been conceived within the SciPy ecosystem.
In a typical work flow, a collection of texts is first transformed
into a Corpus object, whose underlying data structures are NumPy
numerical arrays. The user may then feed a Corpus object to one of the
model classes, which contain the algorithms, implemented in NumPy,
SciPy and IPython.parallel, for training models such as TF, TFIDF,
LSA, BEAGLE or LDA. Finally, the user may examine the
results with a Viewer class specialized to a particular model type. A
Viewer object contains a variety of methods for analysis and
visualization and achieves its full functionality within an IPython
notebook session extended with matplotlib and scikit-learn.

This documentation site is in its early days. We'll soon have
documentation for :mod:`vsm` generated from docstrings and example
IPython notebooks.

.. toctree::
   :maxdepth: 2

   demos
   auto
   
..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`



..
   Acknowledgements
   ================
