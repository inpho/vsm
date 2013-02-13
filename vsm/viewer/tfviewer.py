import numpy as np

from vsm import (enum_sort as _enum_sort_, 
                 map_strarr as _map_strarr_)

from vsm.linalg import row_norms as _row_norms_

from vsm.viewer import (
    def_label_fn as _def_label_fn_,
    res_word_type as _res_word_type_)

from similarity import (
    sim_word_word as _sim_word_word_,
    sim_doc_doc as _sim_doc_doc_,
    simmat_words as _simmat_words_,
    simmat_documents as _simmat_documents_)

from labeleddata import LabeledColumn as _LabeledColumn_



class TfViewer(object):
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
            self._word_norms_ = _row_norms_(self.model.matrix)            

        return self._word_norms_


    @property
    def _doc_norms(self):
        """
        """
        if self._doc_norms_ is None:
            self._doc_norms_ = _row_norms_(self.model.matrix.T)

        return self._doc_norms_


    def sim_word_word(self, word_or_words, weights=None, 
                      filter_nan=True, print_len=10, as_strings=True):
        """
        """
        return _sim_word_word_(self.corpus, self.model.matrix, 
                               word_or_words, weights=weights, 
                               norms=self._word_norms, filter_nan=filter_nan, 
                               print_len=print_len, as_strings=True)


    def sim_doc_doc(self, doc_or_docs, weights=None, print_len=10, 
                    filter_nan=True, label_fn=_def_label_fn_, as_strings=True):
        """
        """
        return _sim_doc_doc_(self.corpus, self.model.matrix,
                             self.model.tok_name, doc_or_docs, weights=weights,
                             norms=self._doc_norms, print_len=print_len,
                             filter_nan=filter_nan, 
                             label_fn=label_fn, as_strings=True)
    

    def simmat_words(self, word_list):
        """
        """
        return _simmat_words_(self.corpus, self.model.matrix, word_list)


    def simmat_docs(self, docs):
        """
        """
        return _simmat_documents_(self.corpus, self.model.matrix,
                                  self.model.tok_name, docs)


    def coll_freq(self, word):
        """
        """
        i,w = _res_word_type_(self.corpus, word)
        row = self.model.matrix.tocsr()[i, :].toarray()
        return row.sum()

    
    def coll_freqs(self, print_len=20, as_strings=True):
        """
        """
        freqs = self.model.matrix.tocsr().sum(1) 
        w_arr = _enum_sort_(freqs.view(np.ndarray)[:, 0])
        
        # Label data
        if as_strings:
            w_arr = _map_strarr_(w_arr, self.corpus.words, k='i', new_k='word')
        w_arr = w_arr.view(_LabeledColumn_)
        w_arr.col_header = 'Collection Frequencies'
        w_arr.subcol_headers = ['Word', 'Counts']
        w_arr.col_len = print_len

        return w_arr



def test_TfViewer():

    from vsm.util.corpustools import random_corpus
    from vsm.model.tf import TfModel

    c = random_corpus(1000, 50, 0, 20, tok_name='document', metadata=True)

    m = TfModel(c, 'document')
    m.train()

    return TfViewer(c, m)
