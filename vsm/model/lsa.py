import numpy as np
from scipy import sparse
from scipy.sparse import linalg as linalgs



class BaseLsaModel(object):

    def __init__(self, context_type, word_matrix=None, 
                 eigenvalues=None, doc_matrix=None):
        """
        """
        self.word_matrix = word_matrix
        self.doc_matrix = doc_matrix
        self.eigenvalues = eigenvalues
        self.context_type = context_type


    def save(self, f):
        """
        Saves model data as a numpy archive file with extension `npz`.
        The keys for the component matrices are `word_matrix`,
        `eigenvalues` and `doc_matrix`.
        
        Parameters
        ----------
        f : str-like or file-like object
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
        
        Parameters
        ----------
        f : str-like or file-like object
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
        print 'Loading model from', f
        arrays_in = np.load(f)
        m = BaseLsaModel(word_matrix=arrays_in['word_matrix'],
                         eigenvalues=arrays_in['eigenvalues'],
                         doc_matrix=arrays_in['doc_matrix'],
                         context_type=arrays_in['context_type'])
        return m



class LsaModel(BaseLsaModel):
    """
    """
    def __init__(self, td_matrix, context_type):
        """
        """
        super(LsaModel, self).__init__(context_type)

        td_matrix = sparse.coo_matrix(td_matrix)
        # Removing infinite values for SVD
        finite_mask = np.isfinite(td_matrix.data)
        coo_in = (td_matrix.data[finite_mask],
                  (td_matrix.row[finite_mask], 
                   td_matrix.col[finite_mask]))
        td_matrix = sparse.coo_matrix(coo_in, shape=td_matrix.shape, 
                                      dtype=np.float64)
        self.td_matrix = td_matrix.tocsr()


    def train(self, k_factors=300):
        """
        """
        if self.td_matrix.shape[0] < k_factors:
            k_factors = self.td_matrix.shape[0] - 1
            
        if self.td_matrix.shape[1] < k_factors:
            k_factors = self.td_matrix.shape[1] - 1

        print 'Performing sparse SVD'
        u, s, v = linalgs.svds(self.td_matrix, k=k_factors)
        self.word_matrix = u
        self.eigenvalues = s
        self.doc_matrix = v



def test_LsaModel():

    from vsm.util.corpustools import random_corpus
    from vsm.model.tf import TfModel
    from vsm.model.tfidf import TfIdfModel

    c = random_corpus(10000, 1000, 0, 30, context_type='document')

    tf = TfModel(c, 'document')
    tf.train()

    tfidf = TfIdfModel(tf.matrix, 'document')
    tfidf.train()

    m = LsaModel(tfidf.matrix, 'document')
    m.train()

    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m.save(tmp.name)
        tmp.close()
        m1 = BaseLsaModel.load(tmp.name)
        assert (m.word_matrix == m1.word_matrix).all()
        assert (m.eigenvalues == m1.eigenvalues).all()
        assert (m.doc_matrix == m1.doc_matrix).all()
    
    finally:
        os.remove(tmp.name)

    return m
