import numpy as np
from scipy import sparse

from vsm import model



class TfModel(model.Model):
    """
    """
    def train(self, corpus, tok_name):

        print 'Retrieving tokens'

        if isinstance(c, MaskedCorpus):

            tokens = corpus.view_tokens(tok_name, compress=True)

        else:

            tokens = corpus.view_tokens(tok_name)

        print 'Computing term frequencies'

        data = np.ones_like(corpus.corpus)
        
        row_indices = corpus.corpus

        col_indices = np.empty_like(corpus.corpus)

        j, k = 0, 0

        for i,token in enumerate(tokens):

            k += len(token)

            col_indices[j:k] = i

            j = k

        shape = (corpus.terms.shape[0], len(tokens))

        coo_in = (data, (row_indices, col_indices))

        self.matrix = sparse.coo_matrix(coo_in, shape=shape, dtype=np.int32)
