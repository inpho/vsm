.. _workflow:

===================
:mod:`vsm` Workflow
===================

The basic stages in a :mod:`vsm` workflow are shown below.

+------------------------+----------------------------------------------+
|                        | Converting a textual corpus into a plain text|
| Raw Processing         | file or a directory tree whose leaves are    |
|                        | plain text files.                            |
|                        |                                              |
+------------------------+----------------------------------------------+
|                        | *Words* and various types of *word* contexts |
| :ref:`tokenization`    | such as *sentences*, *paragraphs*, *pages*,  |
|                        | *articles*, *documents*, *abstracts* or      |
|                        | *books* are identified; as well, labels and  |
|                        | other metadata are gathered and associated to|
|                        | contexts.                                    |
|                        |                                              |
+------------------------+----------------------------------------------+
|                        | A tokenized plain-text corpus is encoded as  |
| :ref:`corpus_encoding` | an integer array; contexts are stored as     |
|                        | indices into the corpus array and additional |
|                        | metadata are stored in records associated to |
|                        | the contexts; stoplists and low-frequency    |
|                        | filters are also applied at this stage.      |
|                        |                                              |
|                        |                                              |
+------------------------+----------------------------------------------+
|                        | A :class:`Corpus` object is given as input to|
| :ref:`model_training`  | a mathematical model for representing a      |
|                        | corpus's semantics; provisions for running a |
|                        | parallel implementation of a model (e.g.,    |
|                        | initiating an :ref:`IPython cluster`) may be |
|                        | made.                                        |
|                        |                                              | 
+------------------------+----------------------------------------------+
|                        | A :class:`Corpus` object and an associated   |
| Viewing and Analyzing  | :mod:`model` object are given to a           |
| Results                | :mod:`viewer` class so that the user may, for|
|                        | example, examine the distances between       |
|                        | semantic representations of words, contexts  |
|                        | (documents) or topics; graphical display and |
|                        | notebook creation options are provided by a  |
|                        | :ref:`IPython notebook server`.              |
|                        |                                              |
|                        |                                              |
+------------------------+----------------------------------------------+

:mod:`vsm` is primarily dedicated to providing robust tools for
carrying out the last three stages: corpus encoding, model training
and exploring results. The first two stages, raw-processing,
tokenization and metadata collection, are largely determined by the
specific form of the raw corpus. :mod:`vsm` provides tools for basic
processing of specific corpora as :mod:`extensions`. Various
lightweight convenience functions for tokenization are included in
:mod:`corpus.util`: written in vanilla Python and employing
:ref:`NLTK` word and sentence tokenizers, they perform reasonably on
medium-sized (<100 million word) corpora.

.. toctree::
    :hidden:
    :maxdepth: 1

    .. raw_processing
    tokenization
    corpus_encoding
    model_training
    .. results_viewing
