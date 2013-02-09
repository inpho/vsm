from scipy import sparse
import numpy as np

from vsm.model import BaseModel



# A term may occur in every document. Then the idf of that term will be
# 0; so the tfidf will also be zero.

# A term may occur in no documents at all. This typically happens only
# when that term has been masked. In that case the idf of that term is
# undefined (division by zero). So also the tfidf of that term will be
# undefined.


class TfIdfModel(BaseModel):

    def __init__(self, tf_matrix, tok_name):
        """
        """
        self.tok_name = tok_name
        self.matrix = tf_matrix.copy()
        self.matrix = self.matrix.tocsr()
        self.matrix = self.matrix.astype(np.float64)


    def train(self):
        """
        """
        n_docs = np.float64(self.matrix.shape[1])

        # Suppress division by zero errors
        old_settings = np.seterr(divide='ignore')

        # NOTE: Adding np.inf wherever we have a zero row results in an
        # overly dense sparse matrix. Leaving `zero_rows` in case we
        # still want to know about them in future code versions.
        zero_rows = []

        print 'Computing tf-idfs'
        for i in xrange(self.matrix.indptr.shape[0] - 1):

            start = self.matrix.indptr[i]
            stop = self.matrix.indptr[i + 1]

            if start == stop:
                zero_rows.append(i)
            else:
                row = self.matrix.data[start:stop]
                row *= np.log(n_docs / np.count_nonzero(row))
                start = stop
        
        # Restore default handling of floating-point errors
        np.seterr(**old_settings)



def test_TfIdfModel():

    from vsm.util.corpustools import random_corpus
    from vsm.model.tf import TfModel

    c = random_corpus(1000, 100, 0, 20, tok_name='document', metadata=True)

    tf = TfModel(c, 'document')
    tf.train()

    m = TfIdfModel(tf.matrix, 'document')
    m.train()

    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m.save(tmp.name)
        tmp.close()
        m1 = TfIdfModel.load(tmp.name)
        assert (m.matrix.todense() == m1.matrix.todense()).all()
    
    finally:
        os.remove(tmp.name)

    return m.matrix
