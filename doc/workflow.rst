===================
:mod:`vsm` Workflow
===================

The basic stages in a :mod:`vsm` workflow are shown below.

+---------------------+---------------------------------------+
|                     | Converting a textual corpus into a    |
| Raw-processing      | plain text file or a directory tree   |
|                     | whose leaves are plain text files     |
+---------------------+---------------------------------------+
|                     | *Words* and various types of *word*   |
| Tokenization and    | contexts such as *sentences*,         |
| Metadata Collection | *paragraphs*, *pages*, *articles*,    |
|                     | *documents*, *abstracts* or *books*   |
|                     | are identified; as well, labels and   |
|                     | other metadata are gathered and       |
|                     | associated to contexts                |
+---------------------+---------------------------------------+
|                     | A tokenized plain-text corpus is      |
| :class:`Corpus`     | encoded as an integer array; contexts |
| Generation          | are stored as indices into the corpus |
|                     | array and additional metadata are     |
|                     | stored in records associated to the   |
|                     | contexts; stoplists and low-frequency |
|                     | filters are also applied at this      |
|                     | stage.                                |
+---------------------+---------------------------------------+
|                     | A :class:`Corpus` object is given as  |
| Model training      | input to a mathematical model for     |
|                     | representing a corpus's semantics;    |
|                     | provisions for running a parallel     |
|                     | implementation of a model (e.g.,      |
|                     | initiating an :ref:`IPython cluster`) |
|                     | may be made.                          |
+---------------------+---------------------------------------+ 
| Viewing and         | A :class:`Corpus` object and an       |
| Analyzing Results   | associated :mod:`model` object are    |
|                     | given to a :mod:`viewer` class so that|
|                     | the user may, for example, examine the|
|                     | distances between semantic            |
|                     | representations of words, contexts    |
|                     | (documents) or topics; graphical      |
|                     | display and notebook creation options |
|                     | are provided by a                     |
|                     | :ref:`IPython notebook` server.       |
+---------------------+---------------------------------------+


.. .. toctree::
..     :maxdepth: 2

..     wf_tutorial
..     corpus_util
..     corpusbuilders
..     module_corpus
..     wf_model
..     wf_viewer

