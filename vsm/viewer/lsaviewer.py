import numpy as np

from vsm.linalg import row_norms as _row_norms_

from vsm.viewer import def_label_fn as _def_label_fn_

from similarity import (
    sim_word_word as _sim_word_word_,
    sim_doc_doc as _sim_doc_doc_,
    simmat_terms as _simmat_terms_,
    simmat_documents as _simmat_documents_)



class LsaViewer(object):
    """
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
            self._word_norms_ = _row_norms_(self.model.term_matrix)            

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
        """
        return _sim_word_word_(self.corpus, self.model.term_matrix, 
                               word_or_words, weights=weights, 
                               norms=self._word_norms, filter_nan=filter_nan, 
                               print_len=print_len, as_strings=True)


    def sim_doc_doc(self, doc_or_docs, print_len=10, filter_nan=True,
                    label_fn=_def_label_fn_, as_strings=True):
        """
        """
        return _sim_doc_doc_(self.corpus, self.model.doc_matrix.T,
                             self.model.tok_name, doc_or_docs,
                             norms=self._doc_norms, print_len=print_len,
                             filter_nan=filter_nan, 
                             label_fn=label_fn, as_strings=True)
    

    def simmat_words(self, word_list):
        """
        """
        return _simmat_terms_(self.corpus, self.model.term_matrix, word_list)


    def simmat_docs(self, docs):
        """
        """
        return _simmat_documents_(self.corpus, self.model.doc_matrix.T,
                                  self.model.tok_name, docs)




def test_LsaViewer():

    from vsm.util.corpustools import random_corpus
    from vsm.model.tf import TfModel
    from vsm.model.tfidf import TfIdfModel
    from vsm.model.lsa import LsaModel

    c = random_corpus(10000, 1000, 0, 30, tok_name='document', metadata=True)

    tf = TfModel(c, 'document')
    tf.train()

    tfidf = TfIdfModel(tf.matrix, 'document')
    tfidf.train()

    m = LsaModel(tfidf.matrix, 'document')
    m.train()

    return LsaViewer(c, m)
