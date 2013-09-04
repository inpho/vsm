from vsm.linalg import row_norms as _row_norms_

from vsm.viewer import def_label_fn as _def_label_fn_

from similarity import (
    sim_word_word as _sim_word_word_,
    sim_doc_doc as _sim_doc_doc_,
    simmat_words as _simmat_words_,
    simmat_documents as _simmat_documents_)



class TfIdfViewer(object):
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
                             self.model.context_type, doc_or_docs, weights=weights,
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
                                  self.model.context_type, docs)




def test_TfIdfViewer():

    from vsm.corpus.util import random_corpus
    from vsm.model.tf import TfModel
    from vsm.model.tfidf import TfIdfModel

    c = random_corpus(1000, 100, 0, 20, context_type='document', metadata=True)

    tf = TfModel(c, 'document')
    tf.train()

    m = TfIdfModel(tf.matrix, 'document')
    m.train()

    return TfIdfViewer(c, m)
