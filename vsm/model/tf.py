import numpy as np
from scipy import sparse

from vsm import model
from vsm import corpus as corp


class TfModel(model.Model):
    """
    """
    def train(self, corpus, tok_name):

        print 'Retrieving tokens'

        if isinstance(corpus, corp.MaskedCorpus):

            tokens = corpus.view_tokens(tok_name, compress=True)

            c = corpus.corpus.compressed()

        else:

            tokens = corpus.view_tokens(tok_name)

            c = corpus.corpus

        print 'Computing term frequencies'

        data = np.ones_like(c)
        
        row_indices = c

        col_indices = np.empty_like(c)

        j, k = 0, 0

        for i,token in enumerate(tokens):

            k += len(token)

            col_indices[j:k] = i

            j = k

        shape = (corpus.terms.shape[0], len(tokens))

        coo_in = (data, (row_indices, col_indices))

        self.matrix = sparse.coo_matrix(coo_in, shape=shape, dtype=np.int32)
