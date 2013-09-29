==============
Corpus Objects
==============

Corpus object stores the given text tokenized as words, sentences and paragraphs. 
It stores information that corresponds to the tokenization. 
For example, words, sentences and paragraphs have indices,
pages in a book can have plain text file information, and articles and books
have corresponding titles or raw files/folder metadata.

:class:`Corpus` extends :class:`BaseCorpus`. Specifications are outlined in the classes. Convenient functions for preparing and making corpus with file or directory can be found in :doc:`corpusbuilders`.

.. automodule:: vsm.corpus

.. toctree::
    :maxdepth: 1

    basecorpus
    corpus

.. currentmodule:: vsm.corpus

.. autofunction:: split_corpus


