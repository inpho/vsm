from scipy import sparse
import numpy as np

from vsm.model import BaseModel



class TfIdfModel(BaseModel):
    """
    Transforms a term-frequency model into a term-frequency
    inverse-document-frequency model.

    A TF-IDF model is term frequency model whose rows, corresponding
    to word types, are scaled by IDF values. The idea is that a word
    type which occurs in most of the contexts (i.e., documents) does
    less to distinguish the contexts semantically than does a word
    type which occurs in few of the contexts. The document frequency
    is the number of documents in which a word occurs divided by the
    number of documents. The IDF is the log of the inverse of the
    document frequency.

    As with a term-frequency model, word types correspond to matrix
    rows and contexts correspond to matrix columns.

    The data structure is a sparse float matrix.

    Parameters
    ----------
    tf_matrix : scipy.sparse matrix
        A matrix containing the term-frequency data.
    context_type : string 
        A string specifying the type of context over which the model
        trainer is applied.

    Attributes
    ----------
    corpus : Corpus
        A Corpus object containing the training data
    context_type : string
        A string specifying the type of context over which the model
        trainer is applied.
    matrix : scipy.sparse.coo_matrix
        A sparse matrix in 'coordinate' format that contains the
        frequency counts.

    Methods
    -------
    train
        Computes the IDF values for the input term-frequency matrix,
        scales the rows by these values and stores the results in
        `self.matrix`
    save
        Takes a filename or file object and saves `self.matrix` and
        `self.context_type` in an npz archive.
    load
        Takes a filename or file object and loads it as an npz archive
        into a BaseModel object.

    See Also
    --------
    tf.TfModel
    BaseModel
    scipy.sparse.coo_matrix


    Notes
    -----

    A zero in the matrix might arise in two ways: (1) the word type
    occurs in every document, in which case the IDF value is 0; (2)
    the word type occurs in no document at all, in which case the IDF
    value is actually undefined.

    TODO: Add a parameter to the training to allow for NANs to appear
    in case 2.
    """
    def __init__(self, tf_matrix, context_type):
        """
        """
        self.context_type = context_type
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

    c = random_corpus(1000, 100, 0, 20, context_type='document', metadata=True)

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
