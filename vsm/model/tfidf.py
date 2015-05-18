import numpy as np
from scipy.sparse import csr_matrix

from base import BaseModel


__all__ = [ 'TfIdf' ]


class TfIdf(BaseModel):
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

    :See Also: :class:`vsm.model.TfSeq`, :class:`vsm.model.base`,
        :class:`scipy.sparse.coo_matrix`

    :notes:
        A zero in the matrix might arise in two ways: (1) the word type
        occurs in every document, in which case the IDF value is 0; (2)
        the word type occurs in no document at all, in which case the IDF
        value is undefined.
    """
    def __init__(self, corpus=None, context_type=None, tf_matrix=None):
        """
        Initialize TfIdf.

        :param corpus: A Corpus object containing the training data.
        :type corpus: Corpus
    
        :param context_type: A string specifying the type of context over
            which the model trainer is applied.
        :type context_type: string 

        :param tf_matrix: A matrix containing the term-frequency data.
        :type tf_matrix: scipy.sparse matrix
        """

        self.context_type = context_type
        if corpus is not None:
            self.corpus = corpus.corpus
        else:
            self.corpus = []

        if tf_matrix is None:
            self.matrix = csr_matrix([], dtype=np.float64)
        else:
            self.matrix = tf_matrix.copy()
            self.matrix = self.matrix.tocsr()
            self.matrix = self.matrix.astype(np.float64)

        self.undefined_rows = []


    def train(self):
        """
        Computes the IDF values for the input term-frequency matrix,
        scales the rows by these values and stores the results in
        `self.matrix`.
        """
        if self.matrix.size > 0:
            n_docs = np.float64(self.matrix.shape[1])
            
            for i in xrange(self.matrix.indptr.shape[0] - 1):

                start = self.matrix.indptr[i]
                stop = self.matrix.indptr[i + 1]

                if start == stop:
                    self.undefined_rows.append(i)
                else:
                    row = self.matrix.data[start:stop]
                    row *= np.log(n_docs / np.count_nonzero(row))
                    start = stop

    @staticmethod
    def from_tf(tf_model):
        """
        Takes a `Tf` model object and generates a `TfIdf` model.
        """
        model = TfIdf(tf_matrix=tf_model.matrix)
        model.corpus = tf_model.corpus
        model.context_type = tf_model.context_type
        return model
