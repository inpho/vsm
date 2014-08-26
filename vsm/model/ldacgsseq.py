import numpy as np
import time
from vsm.split import split_corpus
from ldafunctions import *


__all__ = [ 'LdaCgsSeq' ]



class LdaCgsSeq(object):
    """
    """
    def __init__(self, corpus=None, context_type=None,
                 K=20, alpha=[], beta=[]):
        """
        Initialize LdaCgsSeq.

        :param corpus: Source of observed data.
        :type corpus: `Corpus`
    
        :param context_type: Name of tokenization stored in `corpus` whose tokens
            will be treated as documents.
        :type context_type: string, optional

        :param K: Number of topics. Default is `20`.
        :type K: int, optional
    
        :param top_prior: Topic priors. Default is 0.01 for all topics.
        :type top_prior: list, optional
    
        :param ctx_prior: Context priors. Default is a flat prior of 0.01 
            for all contexts.
        :type ctx_prior: list, optional
        """

        self.context_type = context_type
        self.K = K

        if corpus:
            self.corpus = corpus.corpus
            self.V = corpus.words.size
            i = corpus.context_types.index(self.context_type)
            self.Z_indices = corpus.context_data[i]['idx']
            self.Z_flat = np.zeros_like(self.corpus, dtype=np.int)
        else:
            self.V = 0
            self.corpus = []
            self.Z_indices = []
            self.Z_flat = []

        priors = init_priors(self.V, K, beta, alpha)
        self.beta, self.alpha = priors

        # Word by topic matrix
        self.word_top = (np.zeros((len(self.beta), K), dtype=np.float)
                         + self.beta)
        # Inverse topic sums, initialized to the inverse sums over the
        # topic priors
        self.inv_top_sums = (1. / (np.ones(K, dtype=np.float)
                                   * self.beta.sum()))
        # Topic by document matrix
        self.top_doc = (np.zeros((len(self.alpha), len(self.docs)),
                                 dtype=np.float) + self.alpha)

        self.iteration = 0
        self.log_probs = []


    @property
    def Z(self):
        return split_corpus(self.Z_flat, self.Z_indices)


    @property
    def docs(self):
        return split_corpus(self.corpus, self.Z_indices)


    def train(self, n_iterations=100, verbose=1, random_state=None):

        if verbose > 0:
            print ('Begin LDA training for {0} iterations'\
                   .format(n_iterations))
            start = time.time()
            t = start


        # Training loop
        stop = self.iteration + n_iterations
        for itr in xrange(self.iteration, stop):

            results = cgs_update(self.iteration, self.docs, self.word_top,
                                 self.inv_top_sums, self.top_doc, self.Z, 
                                 random_state=random_state)

            lp = results[4]
            self.log_probs.append((self.iteration, lp))

            if verbose > 0:
                itr_time = np.around(time.time()-t, decimals=1)
                t = time.time()
                if verbose > 1 or itr==stop-1:
                    print ('Iteration {0} complete: log_prob={1}, time={2}'
                           .format(self.iteration, lp, itr_time))
            self.iteration += 1

        if verbose > 1:
            print '-'*60, ('\n\nWalltime per iteration: {0} seconds'
                           .format(np.around((t-start)/n_iterations, decimals=2)))


    @staticmethod
    def load(filename):
        return load_lda(filename, LdaCgsSeq)


    def save(self, filename):
        save_lda(self, filename)
