import numpy as np

from vsm import viewer as vw
import similarity



class LsaViewer(vw.Viewer):
    """

    `svd_matrices` expects the triple of arrays (u, s, v) output by an SVD.

    `matrix` stores the triple of arrays (u, s, v) output by an SVD.


    """
    def __init__(self,
                 corpus=None,
                 svd_matrices=None,
                 tok_name=None):
        
        self.corpus = corpus
        self.matrix = svd_matrices
        self.tok_name = tok_name
        
        self._term_matrix = None
        self._ev_matrix = None
        self._doc_matrix = None
        self._term_norms = None
        self._doc_norms = None



    @property
    def term_matrix(self):

        if self._term_matrix is None:
            
            self._term_matrix = np.dot(self.matrix[0], self.ev_matrix)

        return self._term_matrix



    @property
    def ev_matrix(self):

        if self._ev_matrix is None:
            
            self._ev_matrix = np.diag(self.matrix[1])

        return self._ev_matrix

    

    @property
    def doc_matrix(self):

        if self._doc_matrix is None:
            
            self._doc_matrix = np.dot(self.matrix[2], self.ev_matrix)

        return self._doc_matrix



    @property
    def term_norms(self):

        if self._term_norms is None:

            self._term_norms = similarity.row_norms(self.term_matrix)            

        return self._term_norms



    @property
    def doc_norms(self):

        if self._doc_norms is None:

            self._doc_norms = similarity.row_norms(self.doc_matrix)            

        return self._doc_norms


    
    def similar_terms(self, term, filter_nan=False):

        return vw.similar_terms(self.corpus,
                                self.term_matrix,
                                term,
                                norms=self.term_norms,
                                filter_nan=filter_nan)



    def similar_documents(self, document, filter_nan=False):

        return vw.similar_documents(self.corpus,
                                    self.term_matrix,
                                    document,
                                    norms=self.doc_norms,
                                    filter_nan=filter_nan)



    def simmat_terms(self, term_list):

        return vw.simmat_terms(self.corpus,
                               self.term_matrix,
                               term_list)



    def simmat_documents(self, document_list):

        return vw.simmat_documents(self.corpus,
                                   self.doc_matrix,
                                   document_list)
