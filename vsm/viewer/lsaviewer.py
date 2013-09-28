import numpy as np

from vsm.linalg import row_norms as _row_norms_

from vsm.viewer import def_label_fn as _def_label_fn_

from similarity import (
    sim_word_word as _sim_word_word_,
    sim_doc_doc as _sim_doc_doc_,
    simmat_words as _simmat_words_,
    simmat_documents as _simmat_documents_)



class LsaViewer(object):
    """
    A class for viewing LSA model.

    :param corpus: Source of observed data.
    :type corpus: Corpus

    :param model: An LSA model.
    :type model: Lsa object.

    :attributes:
        * **corpus** (Corpus object) - `corpus`
        * **model** (Tf object) - `model`
        * **_words_norms_**
        * **_doc_norms_**

    :methods:
        * **sim_word_word**
            Returns words sorted by the cosine values between a word or list
            of words and every word.
        * **sim_doc_doc**
            Computes and sorts the cosine similarity values between a
            document or list of documents and every document.
        * **simmat_words**
            Calculates the similarity matrix for a given list of words.
        * **simmat_docs**
            Calculates the similarity matrix for a given list of documents.

    :See Also: :mod:`vsm.model.lsa`
    """
    def __init__(self, corpus, model):
        """
        """
        self.corpus = corpus
        self.model = model
        self._word_norms_ = None
        self._doc_norms_ = None


    @property
    def _word_norms(self):
        """
        """
        if self._word_norms_ is None:
            self._word_norms_ = _row_norms_(self.model.word_matrix)            

        return self._word_norms_


    @property
    def _doc_norms(self):
        """
        """
        if self._doc_norms_ is None:
            self._doc_norms_ = _row_norms_(self.model.doc_matrix)

        return self._doc_norms_


    def sim_word_word(self, word_or_words, weights=None, 
                      filter_nan=True, print_len=10, as_strings=True):
        """
        A wrapper of `sim_word_word` in similarity.py

        :param word_or_words: Query word(s) to which cosine values are calculated.
        :type word_or_words: string or list of strings
        
        :param weights: Specify weights for each query word in `word_or_words`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional
        
        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional

        :param print_len: Number of words printed by pretty-printing function
            Default is 10.
        :type print_len: int, optional

        :param as_strings: If `True`, returns a list of words as strings rather
            than their integer representations. Default is `True`.
        :type as_strings: boolean, optional
        
        :returns: w_arr : a LabeledColumn object
            A 2-dim array containing words and their cosine values to 
            `word_or_words`. 
        
        :See Also: :meth:`vsm.viewer.similarity.sim_word_word`
        """
        return _sim_word_word_(self.corpus, self.model.word_matrix, 
                               word_or_words, weights=weights, 
                               norms=self._word_norms, filter_nan=filter_nan, 
                               print_len=print_len, as_strings=True)


    def sim_doc_doc(self, doc_or_docs, weights=None, print_len=10, 
                    filter_nan=True, label_fn=_def_label_fn_, as_strings=True):
        """
        :param doc_or_docs: Query document(s) to which cosine values
            are calculated
        :type doc_or_docs: string/integer or list of strings/integers
        
        :param weights: Specify weights for each query doc in `doc_or_docs`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional
        
        :param print_len: Number of words printed by pretty-printing function.
            Default is 10.
        :type print_len: int, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional
 
        :param label_fn: A function that defines how documents are represented.
            Default is def_label_fn which retrieves the labels from corpus metadata.
        :type label_fn: string, optional
        
        :param as_strings: If `True`, returns a list of words rather than
            their integer representations. Default is `True`.
        :type as_strings: boolean, optional

        :returns: w_arr : a LabeledColumn object
            A 2-dim array containing documents and their cosine values to 
            `doc_or_docs`. 
        
        :See Also: :meth:`vsm.viewer.similarity.sim_doc_doc`
        """
        return _sim_doc_doc_(self.corpus, self.model.doc_matrix.T,
                             self.model.context_type, doc_or_docs, weights=weights,
                             norms=self._doc_norms, print_len=print_len,
                             filter_nan=filter_nan, 
                             label_fn=label_fn, as_strings=True)
    

    def simmat_words(self, word_list):
        """
        Calculates the similarity matrix for a given list of words.

        :param word_list: A list of words whose similarity matrix is to be
            computed.
        :type word_list: list

        :returns: an IndexedSymmArray object
            n x n matrix containing floats where n is the number of words
            in `word_list`.
        
        :See Also: :meth:`vsm.viewer.similarity.simmat_words`
        """
        return _simmat_words_(self.corpus, self.model.word_matrix, word_list)


    def simmat_docs(self, docs):
        """
        Calculates the similarity matrix for a given list of documents.

        :param docs: A list of documents whose similarity matrix is to be computed.
            Default is all the documents in the model.
        :type docs: list, optional
        
        :returns: an IndexedSymmArray object
            n x n matrix containing floats where n is the number of documents.           
            considered.

        :See Also: :meth:`vsm.viewer.similarity.simmat_docs`
        """
        return _simmat_documents_(self.corpus, self.model.doc_matrix.T,
                                  self.model.context_type, docs)




def test_LsaViewer():

    from vsm.corpus.util import random_corpus
    from vsm.model.tf import TfModel
    from vsm.model.tfidf import TfIdfModel
    from vsm.model.lsa import LsaModel

    c = random_corpus(10000, 1000, 0, 30, context_type='document', metadata=True)

    tf = TfModel(c, 'document')
    tf.train()

    tfidf = TfIdfModel(tf.matrix, 'document')
    tfidf.train()

    m = LsaModel(tfidf.matrix, 'document')
    m.train()

    return LsaViewer(c, m)
