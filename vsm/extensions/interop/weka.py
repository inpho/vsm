"""
`vsm.extensions.interop.weka`

Module containing functions for import/export between VSM and Weka,
a collection of machine learning algorithms for data mining tasks
implemented in Java. Weka is available at:
`<http://www.cs.waikato.ac.nz/ml/weka/>`_

This module imports and exports corpora to the `ARFF format`_ used 
by Weka. ARFF files can then be used for `text categorization with Weka`_.


.. _ARFF format: https://weka.wikispaces.com/ARFF
.. _text categorization with Weka: 
    https://weka.wikispaces.com/Text+categorization+with+Weka

"""
import os
import os.path

from scipy.stats import itemfreq
import numpy as np

from vsm.extensions.corpusbuilders import corpus_fromlist


def export_corpus(corpus, outfolder, context_type='document'):
    """
    Converts a vsm.corpus.Corpus object into a Weka-compatible `ARFF file`_.

    :param corpus: VSM Corpus object to convert to lda-c file
    :type corpus: vsm.corpus.Corpus

    :param outfolder: Directory to output "vocab.txt" and "corpus.dat"
    :type string: path

    .. _ARFF file: https://weka.wikispaces.com/ARFF
    """
    pass


def import_corpus(corpusfilename, vocabfilename, context_type='document'):
    """
    Converts an lda-c compatible data file into a VSM Corpus object.

    :param corpusfilename: path to corpus file, as defined in lda-c
    documentation.
    :type string:

    :param vocabfilename: path to vocabulary file, one word per line
    :type string:
    """
    pass


def import_model(filename):
    pass


def export_model(filename):
    pass
