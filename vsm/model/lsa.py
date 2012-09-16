import numpy as np

from scipy import sparse
from scipy.sparse import linalg as linalgs

from vsm import model
from vsm.model import tfidf

# TODO: train needs either 1) a matrix or 2) corpus and tok_name;
# provide feedback to this effect

class LsaModel(model.Model):
    """
    """
    def train(self,
              corpus=None,
              tok_name=None,
              td_matrix=None,
              k_factors=300):

        if td_matrix is None:

            tfidf_model = tfidf.TfIdfModel()

            tfidf_model.train(corpus, tok_name)
            
            td_matrix = tfidf_model.matrix

            del tfidf_model

        if td_matrix.shape[0] < k_factors:

            k_factors = td_matrix.shape[0] - 1
            
        if td_matrix.shape[1] < k_factors:

            k_factors = td_matrix.shape[1] - 1

        td_matrix = sparse.csc_matrix(td_matrix, dtype=np.float64)

        print 'Performing sparse SVD'

        u, s, v = linalgs.svds(td_matrix, k=k_factors)

        self.matrix = np.float32(u), np.float32(s), np.float32(v)

        

    @property
    def term_matrix(self):

        return self.matrix[0]



    @property
    def eigenvalues(self):

        return self.matrix[1]



    @property
    def doc_matrix(self):

        return self.matrix[2]
        


    def save(self, file):
        """
        Saves matrix data as a numpy archive file with extension
        `npz`. The keys for the component matrices are `term_matrix`,
        `eigenvalues` and `doc_matrix`.
        
        Parameters
        ----------
        file : str-like or file-like object
            Designates the file to which to save data. See
            `numpy.savez` for further details.
            
        Returns
        -------
        None

        See Also
        --------
        LsaModel.load
        numpy.savez
        """
        print 'Saving matrices as', file
        
        arrays_out = dict()
        
        arrays_out['term_matrix'] = self.term_matrix
        
        arrays_out['eigenvalues'] = self.eigenvalues

        arrays_out['doc_matrix'] = self.doc_matrix

        np.savez(file, **arrays_out)



    @staticmethod
    def load(file):
        """
        Loads LSA matrix data from a numpy archive file with extension
        `npz`. The expected keys for the component matrices are
        `term_matrix`, `eigenvalues` and `doc_matrix`.
        
        Parameters
        ----------
        file : str-like or file-like object
            Designates the file from which to load data. See
            `numpy.load` for further details.
            
        Returns
        -------
        None

        See Also
        --------
        LsaModel.save
        numpy.load
        """
        print 'Loading LSA matrices from', file

        arrays_in = np.load(file)

        return (arrays_in['term_matrix'],
                arrays_in['eigenvalues'],
                arrays_in['doc_matrix'])
