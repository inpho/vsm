import numpy as np
from scipy import sparse


__all__ = [ 'Lsa' ]


class Lsa(object):
    """
    """
    
    def __init__(self, corpus=None, context_type=None, td_matrix=None):
        """
        Initialize Lsa.

        :param corpus: A Corpus object containing the training data.
        :type corpus: Corpus, optional

        :param context_type: Name of tokenization whose tokens will be
            treated as documents. Default is `None`.
        :type context_type: string, optional

        :param td_matrix: Term-Document matrix. Default is `None`.
        :type td_matrix: np.array, optional
        """

        self.word_matrix = None
        self.doc_matrix = None
        self.eigenvalues = None
        self.context_type = context_type
        if corpus is not None:
            self.corpus = corpus.corpus
        else:
            self.corpus = []

        if td_matrix is None:
            self.td_matrix = np.array([])
        else:
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
        Trains the model.

        :param k_factors: Default is 300.
        :type k_factors: int, optional
        """
        from scipy.sparse import linalg as linalgs

        u,s,v = np.array([]), np.array([]), np.array([])

        if self.td_matrix.size > 0:
            s = min(self.td_matrix.shape)
            if s < k_factors:
                k_factors = s - 1

            # print 'Performing sparse SVD'
            u, s, v = linalgs.svds(self.td_matrix, k=k_factors)

        indices = s.argsort()[::-1]
        self.word_matrix = u[:, indices]
        self.eigenvalues = s[indices]
        self.doc_matrix = v[indices, :]


    def save(self, f):
        """
        Saves model data as a numpy archive file with extension `npz`.
        The keys for the component matrices are `word_matrix`,
        `eigenvalues` and `doc_matrix`.
        
        :param f: Designates the file to which to save data. See
            `numpy.savez` for further details.
        :type f: str-like or file-like object
            
        :See Also: :meth:`numpy.savez`
        """
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
            
        :returns: a saved Lsa model.

        :See Also: :meth:`numpy.load`
        """
        arrays_in = np.load(f)
        m = Lsa(context_type=arrays_in['context_type'])
        m.word_matrix=arrays_in['word_matrix']
        m.eigenvalues=arrays_in['eigenvalues']
        m.doc_matrix=arrays_in['doc_matrix']
        return m

    @staticmethod
    def from_tf(tf_model):
        """
        Takes a `Tf` model object and generates a `TfIdf` model.
        """
        model = Lsa(td_matrix=tf_model.matrix)
        model.corpus = tf_model.corpus
        model.context_type = tf_model.context_type
        return model
    
    @staticmethod
    def from_tfidf(tfidf_model):
        """
        Takes a `Tf` model object and generates a `TfIdf` model.
        """
        model = Lsa(td_matrix=tfidf_model.matrix)
        model.corpus = tfidf_model.corpus
        model.context_type = tfidf_model.context_type
        return model
