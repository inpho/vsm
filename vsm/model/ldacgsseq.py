import numpy as np
import time

from ldafunctions import(
    init_priors as _init_priors_,
    cgs_update as _update_,
    load_lda as _load_lda_,
    save_lda as _save_lda_)


class LdaCgsSeq(object):
    """
    """
    def __init__(self, corpus=None, context_type=None,
                 K=100, alpha=[], beta=[]):

        self.context_type = context_type
        self.K = K

        if corpus:
            self._load_corpus(corpus)
        else:
            self.docs = []
            self.V = 0

        priors = _init_priors_(self.V, K, beta, alpha)
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
        self.Z=[np.zeros_like(doc) for doc in self.docs]

        self.iteration = 0
        self.log_probs = []

    def _load_corpus(self, corpus):
        self.docs = corpus.view_contexts(self.context_type)
        self.V = corpus.words.size

    def train(self, n_iterations=100, verbose=1, random_state=None):

        if verbose > 0:
            print ('Begin LDA training for {0} iterations'\
                   .format(n_iterations))
            start = time.time()
            t = start


        # Training loop
        stop = self.iteration + n_iterations
        for itr in xrange(self.iteration, stop):

            results = _update_(self.iteration, self.docs, self.word_top,
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

    def query_sample(self, doc, n_iterations, random_state=None):

        sampler = query_sampler(doc, self)
        sampler.train(n_iterations, random_state=random_state)

        return sampler

    @staticmethod
    def load(filename, corpus=None):
        return _load_lda_(filename, LdaCgsSeq, corpus=corpus)

    def save(self, filename):
        _save_lda_(self, filename)


def query_sampler(doc, m):
    """
    Takes an LDA model object `m`, a fresh document (assumed to be
    from the same corpus) `doc` and returns a ``query sampler'' as an
    LDA model object. Note that this sampler contains a single
    document and so a Kx1 matrix as its topic-document matrix and Z
    variables corresponding only to this document; while it contains
    the word-topic matrix from m (so the counts in `Z` and `docs` do
    not correspond to the counts in `word_top`).
    """
    sampler = LdaCgsSeq()
    sampler.docs = [np.array(doc, dtype=np.int)]
    sampler.V = len(doc)
    sampler.context_type = m.context_type
    sampler.K = m.K
    sampler.beta = m.beta.copy()
    sampler.alpha = m.alpha.copy()
    sampler.word_top = m.word_top.copy()
    sampler.Z = [np.zeros_like(doc)]
    sampler.top_doc = np.zeros((m.K, 1), dtype=np.float) + m.alpha
    
    return sampler
