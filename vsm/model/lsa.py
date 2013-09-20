import numpy as np
from scipy import sparse
from scipy.sparse import linalg as linalgs



class BaseLsaModel(object):

    def __init__(self, context_type=None, word_matrix=None, 
                 eigenvalues=None, doc_matrix=None):

        self.word_matrix = word_matrix
        self.doc_matrix = doc_matrix
        self.eigenvalues = eigenvalues
        self.context_type = context_type


    def save(self, f):
        """
        Saves model data as a numpy archive file with extension `npz`.
        The keys for the component matrices are `word_matrix`,
        `eigenvalues` and `doc_matrix`.
        
        :param f: Designates the file to which to save data. See
            `numpy.savez` for further details.
        :type f: str-like or file-like object
            
        :returns: None

        :See Also: :meth: LsaModel.load, :meth: numpy.savez
        """
        print 'Saving model as', f
        arrays_out = dict()
        arrays_out['word_matrix'] = self.word_matrix
        arrays_out['eigenvalues'] = self.eigenvalues
        arrays_out['doc_matrix'] = self.doc_matrix
        arrays_out['context_type'] = self.context_type
        np.savez(f, **arrays_out)


    @staticmethod
    def load(f):
        """
        Loads LSA model data from a numpy archive file with extension
        `npz`. The expected keys for the component matrices are
        `word_matrix`, `eigenvalues` and `doc_matrix`.
        
        :param f: Designates the file from which to load data. See
            `numpy.load` for further details.
        :type f: str-like or file-like object
            
        :returns: None

        :See Also: :meth: LsaModel.save, :meth: numpy.load
        """
        print 'Loading model from', f
        arrays_in = np.load(f)
        m = BaseLsaModel(word_matrix=arrays_in['word_matrix'],
                         eigenvalues=arrays_in['eigenvalues'],
                         doc_matrix=arrays_in['doc_matrix'],
                         context_type=arrays_in['context_type'])
        return m



class Lsa(BaseLsaModel):
    """
    """
    def __init__(self, td_matrix=np.array([]), context_type=None):

        super(Lsa, self).__init__(context_type)

        if td_matrix.size > 0:
            td_matrix = sparse.coo_matrix(td_matrix)
            
            # Removing infinite values for SVD
            finite_mask = np.isfinite(td_matrix.data)
            coo_in = (td_matrix.data[finite_mask],
                      (td_matrix.row[finite_mask], 
                       td_matrix.col[finite_mask]))

            td_matrix = sparse.coo_matrix(coo_in, shape=td_matrix.shape, 
                                          dtype=np.float64)
            self.td_matrix = td_matrix.tocsr()
        else:
            self.td_matrix = np.array([])


    def train(self, k_factors=300):
        """
        :param k_factors: Default is 300.
        :type k_factors: int, optional

        :returns: None

        """

        u,s,v = np.array([]), np.array([]), np.array([])

        if self.td_matrix.size > 0:
            s = min(self.td_matrix.shape)
            if s < k_factors:
                k_factors = s - 1

            print 'Performing sparse SVD'
            u, s, v = linalgs.svds(self.td_matrix, k=k_factors)

        self.word_matrix = u
        self.eigenvalues = s
        self.doc_matrix = v
