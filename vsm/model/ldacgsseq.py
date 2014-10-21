import numpy as np
import time
from vsm.split import split_corpus
from ldafunctions import *
from _cgs_update import cgs_update


__all__ = [ 'LdaCgsSeq' ]



class LdaCgsSeq(object):
    """
    An implementation of LDA using collapsed Gibbs sampling.
    """
    def __init__(self, corpus=None, context_type=None,
                 K=20, V=0, alpha=[], beta=[]):
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
            self.V = V
            self.indices = []
            self.corpus = []

        self.Z = np.zeros_like(self.corpus, dtype=np.int)

        priors = init_priors(self.V, self.K, beta, alpha)
        self.beta, self.alpha = priors

        self.word_top = (np.zeros((self.V, self.K), dtype=np.float)
                         + self.beta)
        if self.V==0:
            self.inv_top_sums = np.inf
        else:
            self.inv_top_sums = 1. / self.word_top.sum(0)
        self.top_doc = (np.zeros((self.K, len(self.indices)),
                                 dtype=np.float) + self.alpha)

        self.iteration = 0
        self.log_probs = []


    @property
    def Z_split(self):
        return split_corpus(self.Z, self.indices)


    @property
    def docs(self):
        return split_corpus(self.corpus, self.indices)


    def _compute_word_top(self):
        self.word_top = compute_word_top(self.docs, self.Z_split, self.K, 
                                         self.V, self.beta)


    def _compute_top_doc(self):
        self.top_doc = compute_top_doc(self.Z_split, self.K, self.alpha)


    def _compute_log_prob(self, increment=False):
        log_prob = compute_log_prob(self.docs, self.Z_split, 
                                    self.word_top, self.top_doc)
        if increment:
            self.log_probs.append((self.iteration, log_prob))
            self.iteration += 1
        else:
            return log_prob


    def train(self, n_iterations=100, verbose=1, seed=None):
        """
        Takes an optional argument, `n_iterations` and updates the model
        `n_iterations` times.

        :param n_iterations: Number of iterations. Default is 100.
        :type n_iterations: int, optional

        :param verbose: If 1, current number of iterations
            are printed out to notify the user. Default is 1.
        :type verbose: int, optional

        :param seeds: An arbitrary starting point. If `None` it will
            start at a random seed. Default is `None`.
        :type seeds: double, optional
        """
        random_state = np.random.RandomState(seed)
        mtrand_state = random_state.get_state()

        if verbose > 0:
            print ('Begin LDA training for {0} iterations'\
                   .format(n_iterations))
            start = time.time()
            t = start

        # Training loop
        stop = self.iteration + n_iterations
        for itr in xrange(self.iteration, stop):

            results = cgs_update(self.iteration, self.corpus, self.word_top,
                                 self.inv_top_sums, self.top_doc, self.Z, 
                                 self.indices, mtrand_state[0], 
                                 mtrand_state[1], mtrand_state[2], 
                                 mtrand_state[3], mtrand_state[4])

            lp = results[4]
            self.log_probs.append((self.iteration, lp))

            if verbose > 0:
                itr_time = np.around(time.time()-t, decimals=1)
                t = time.time()
                if verbose > 1 or itr==stop-1:
                    print ('Iteration {0} complete: log_prob={1}, time={2}'
                           .format(self.iteration, lp, itr_time))
            self.iteration += 1

            mtrand_state = results[5:]

        if verbose > 1:
            print '-'*60, ('\n\nWalltime per iteration: {0} seconds'
                           .format(np.around((t-start)/n_iterations, decimals=2)))


    @staticmethod
    def load(filename):
        """
        A static method for loading a saved LdaCgsMulti model.

        :param filename: Name of a saved model to be loaded.
        :type filename: string

        :returns: m : LdaCgsMulti object

        :See Also: :class:`numpy.load`
        """
        return load_lda(filename, LdaCgsSeq)


    def save(self, filename):
        """
        Saves the model in an `.npz` file.

        :param filename: Name of a saved model to be loaded.
        :type filename: string

        :See Also: :class:`numpy.savez`
        """
        save_lda(self, filename)



#################################################################
#                            Demos
#################################################################


def demo_LdaCgsSeq(doc_len=500, V=100000, n_docs=100,
                   K=20, n_iterations=5, 
                   corpus_seed=None, model_seed=None):

    from vsm.extensions.corpusbuilders import random_corpus
    
    print 'Words per document:', doc_len
    print 'Words in vocabulary:', V
    print 'Documents in corpus:', n_docs
    print 'Number of topics:', K
    print 'Iterations:', n_iterations

    c = random_corpus(n_docs*doc_len, V, doc_len, doc_len+1, seed=corpus_seed)
    m = LdaCgsSeq(c, 'document', K=K)
    m.train(n_iterations=n_iterations, verbose=2, seed=model_seed)

    return m
