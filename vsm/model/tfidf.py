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
    def train(self, corpus=None, tok_name=None, tf_matrix=None):

        if tf_matrix is None:

            tf_model = tf.TfModel()

            tf_model.train(corpus, tok_name)

            self.matrix = tf_model.matrix

            del tf_model

        else:

            self.matrix = tf_matrix

        if sparse.issparse(tf_matrix):

            self.matrix = self.matrix.tocsr()

        del tf_matrix

        self.matrix = self.matrix.astype(np.float32)

        n_docs = np.float32(self.matrix.shape[1])

        print 'Computing tf-idfs'

        # Suppress division by zero errors

        old_settings = np.seterr(divide='ignore')

        zero_rows = []

        for i in xrange(self.matrix.indptr.shape[0] - 1):

            start = self.matrix.indptr[i]

            stop = self.matrix.indptr[i + 1]
            
            if start == stop:

                zero_rows.append(i)

            else:

                row = self.matrix.data[start:stop]
                
                row *= np.log(n_docs / np.count_nonzero(row))
                
                start = stop

        # NOTE: Adding np.inf wherever we have a zero row results in a
        # too dense sparse matrix. Leaving `zero_rows` in case we
        # still want to know about them in future code versions.
        
        # Restore default handling of floating-point errors

        np.seterr(**old_settings)
