import numpy as np

import vsm.viewer as vw
import similarity



class TfViewer(vw.Viewer):

    def __init__(self,
                 corpus=None,
                 matrix=None,
                 tok_name=None):
        
        super(TfViewer, self).__init__(corpus=corpus,
                                       matrix=matrix,
                                       tok_name=tok_name)
        
        self._term_norms = None
        self._doc_norms = None


        
    @property
    def term_norms(self):

        if self._term_norms is None:

            self._term_norms = similarity.row_norms(self.matrix)

        return self._term_norms



    @property
    def doc_norms(self):

        if self._doc_norms is None:

            self._doc_norms = similarity.col_norms(self.matrix)            

        return self._doc_norms



    def similar_terms(self, term, filter_nan=True, rem_masked=True):

        return vw.similar_terms(self.corpus,
                                self.matrix,
                                term,
                                norms=self.term_norms,
                                filter_nan=filter_nan,
                                rem_masked=rem_masked)



    def mean_similar_terms(self, query, filter_nan=True, rem_masked=True):

        return vw.mean_similar_terms(self.corpus,
                                     self.matrix,
                                     query,
                                     norms=self.term_norms,
                                     filter_nan=filter_nan,
                                     rem_masked=rem_masked)



    def similar_documents(self, document, filter_nan=False):

        return vw.similar_terms(self.corpus,
                                self.matrix,
                                document,
                                norms=self.doc_norms,
                                filter_nan=filter_nan)


    
    def simmat_terms(self, term_list):

        return vw.simmat_terms(self.corpus, self.matrix, term_list)



    def simmat_documents(self, document_list):

        return vw.simmat_documents(self.corpus, self.matrix, document_list)


    
    def cf(self, term):
        """
        """
        row = self.matrix.tocsr()[term,:]
        
        return row.sum(1)[0, 0]

    
    def cfs(self):
        """
        """
        return np.asarray(self.matrix.tocsr().sum(1)).ravel()
