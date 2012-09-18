from scipy import sparse
import numpy as np

from vsm import model
from vsm.model import tf


"""
A term may occur in every document. Then the idf of that term will be
0; so the tfidf will also be zero.

A term may occur in no documents at all. This typically happens only
when that term has been masked. In that case the idf of that term is
undefined (division by zero). So also the tfidf of that term will be
undefined.
"""


class TfIdfModel(model.Model):
    """
    """
    def train(self, corpus, tok_name=None, tf_matrix=None):

        if tf_matrix is None:

            tf_model = tf.TfModel()

            tf_model.train(corpus, tok_name)

            tf_matrix = tf_model.matrix

            del tf_model

        if sparse.issparse(tf_matrix):

            tf_matrix = tf_matrix.tocsr()
            
        tf_matrix = tf_matrix.astype(np.float32)

        n_docs = np.float32(tf_matrix.shape[1])

        print 'Computing tf-idfs'

        idfs = np.empty(tf_matrix.shape[0])

        # Suppress division by zero errors

        old_settings = np.seterr(divide='ignore')

        for i in xrange(tf_matrix.shape[0]):

            idfs[i] = np.log(n_docs / tf_matrix[i:i+1, :].nnz)
            
        # Restore default handling of floating-point errors

        np.seterr(**old_settings)

        tf_matrix = tf_matrix.tocoo()

        row = tf_matrix.row.tolist()

        col = tf_matrix.col.tolist()

        data = tf_matrix.data.tolist()

        shape, dtype = tf_matrix.shape, tf_matrix.dtype

        del tf_matrix

        for k,i in enumerate(row):

            idf = idfs[i]

            if np.isfinite(idf):

                data[k] *= idf

        for i,idf in enumerate(idfs):

            if not np.isfinite(idf):

                row.extend([i] * n_docs)

                col.extend(xrange(n_docs))

                data.extend([idf] * n_docs)

        coo_in = (data, (row, col))

        self.matrix = sparse.coo_matrix(coo_in, shape=shape, dtype=dtype)


