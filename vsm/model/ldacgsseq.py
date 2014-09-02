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
    
        :param beta: Topic priors. Default is 0.01 for all words.
        :type beta: list, optional
    
        :param alpha: Document priors. Default is a flat prior of 0.01 
            for all topics.
        :type alpha: list, optional
        """

        self.context_type = context_type
        self.K = K

        if corpus:
            self.V = corpus.words.size
            self.indices = corpus.view_contexts(self.context_type, as_indices=True)
            self.corpus = corpus.corpus
        else:
            self.V = 0
            self.indices = []
            self.corpus = []

        self.Z = np.zeros_like(self.corpus, dtype=np.int)

        priors = init_priors(self.V, self.K, beta, alpha)
        self.beta, self.alpha = priors

        self.word_top = (np.zeros((len(self.beta), self.K), dtype=np.float)
                         + self.beta)
        self.inv_top_sums = (1. / (np.ones(self.K, dtype=np.float)
                                   * self.beta.sum()))
        self.top_doc = (np.zeros((len(self.alpha), len(self.docs)),
                                 dtype=np.float) + self.alpha)

        self.iteration = 0
        self.log_probs = []


    @property
    def Z_split(self):
        return split_corpus(self.Z, indices)


    @property
    def docs(self):
        return split_corpus(self.corpus, self.indices)


    def train(self, n_iterations=100, verbose=1, random_state=None):

        if random_state==None:
            random_state=np.random.RandomState()

        if verbose > 0:
            print ('Begin LDA training for {0} iterations'\
                   .format(n_iterations))
            start = time.time()
            t = start

        # Training loop
        stop = self.iteration + n_iterations
        for itr in xrange(self.iteration, stop):

            results = cgs_update(self.iteration, self.docs, self.word_top,
                                 self.inv_top_sums, self.top_doc, self.Z_split, 
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
