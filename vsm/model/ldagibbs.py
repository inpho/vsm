from sys import stdout
from math import log
import numpy as np




def smpl_cat(d_cum):
    """
    Takes an array of cumurative probability distribution d and returns 
    a sample from the categorical distribution parameterized by d.
    """    
    r = np.random.random() * d_cum[-1]
    return np.searchsorted(d_cum, r)



class LDAGibbs(object):
    """
    An implementation of LDA using collapsed Gibbs sampling.

    References
    ----------    
    Griffiths, Tom. Gibbs sampling in the generative model of Latent Dirichlet Allocation.

    Wang, Yi. Distributed Gibbs Sampling of Latent Topic Models: The Gritty Details.

    Parameters
    ----------
    corpus : Corpus
        Source of observed data
    context_type : string
        Name of tokenization stored in `corpus` whose tokens will be
        treated as documents.
    K : int
        Number of topics. Default is `100`.
    alpha : float
        Parameter for the prior distribution of theta_d. Default is `0.01`.
    beta : float
        Parameter for the prior distribution of phi_d. Default is `0.01`.
    log_prob : boolean
        If `True`, compute the log probabilities of the corpus given
        the values of the latent variables at each iteration and
        records them in `log_prob`. Default is `True`.

    Attributes
    ----------
    W : list of integer arrays
        List of documents, which are extracted from the input Corpus object
    V : int
        Number of unique words in the corpus
    Z : list of integer arrays
        Topic assignments for every word coordinate in the corpus
    iterations : int
        Number of past iterations of the update rule
    doc_top : 2-dim floating point array
        Stores the unnormalized estimated posterior distribution over
        topics for each document in a D x K matrix
    top_word : 2-dim floating point array
        Stores the unnormalized estimated posterior distribution over
        terms for each topic in a K x V matrix
    sum_word_top : 1-dim floating point array
        Stores the sum of words over topics

    Methods
    -------
    train
        Takes an optional argument `itr`, which defaults to 1000, and
        updates the model `itr` times.
    update_z
        Takes a document index `d`, a word index `i` relative to that
        document and a word `w` and updates the model.
    z_dist
        Takes a document index `d` and a word `w` and computes the
        distribution over topics for `w` in `d`
    phi_k
        Takes a topic index `t` and returns the estimated posterior
        distribution over words for `t`
    phi_w
        Takes a word `w` and returns the estimated posterior
        distribution over topics for `w`
    theta_d
        Takes a document index `d` and returns the estimated posterior
        distribution over topics for `d`
    theta_k
        Takes a topic index `t` and returns the estimated posterior
        distribution over documents for `t`
    logp
        Compute the log probability of the corpus `W` given the
        estimated values of the latent variables `phi`, `theta` and
        `Z`

    """
    def __init__(self, corpus, context_type,
                 K=100, alpha = 0.01, beta = 0.01, 
                 log_prob=True):

        self.context_type = context_type
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.W = corpus.view_contexts(context_type)
        self.V = corpus.words.shape[0]
        self.iterations = 0

        if log_prob:
            self.log_prob = []
        
        # Initialize

        self.Z = [np.zeros_like(d) for d in self.W]
        self.doc_top = np.zeros((len(self.W), K)) + alpha
        self.top_word = np.zeros((K, self.V)) + beta
        self.sum_word_top = (self.V * beta) + np.zeros(K)

        for d, doc in enumerate(self.W):
            for i, w in enumerate(doc):
                self.update_z(d, i, w)


    def train(self, itr=1000, verbose=True):

        for t in xrange(self.iterations, self.iterations + itr):

            if verbose:
                stdout.write('\rIteration %d' % t)
                stdout.flush()

            self.iterations += 1
            
            if hasattr(self, 'log_prob'):
                self.log_prob.append((t, self.logp()))

            for d, doc in enumerate(self.W):
                for i, w in enumerate(doc):
                    z = self.Z[d][i]
                    self.doc_top[d, z] -= 1
                    self.top_word[z, w] -= 1
                    self.sum_word_top[z] -= 1
                    self.update_z(d, i, w)

        if verbose:
            stdout.write('\n')



    def z_dist(self, d, w):

        sum_word_top_inv = 1. / self.sum_word_top
        dist = (self.doc_top[d, :] * self.top_word[:, w]  * sum_word_top_inv)
        dist_cum = np.cumsum(dist)
        return dist_cum



    def update_z(self, d, i, w):
        
        z = smpl_cat(self.z_dist(d, w))
        self.doc_top[d, z] += 1
        self.top_word[z, w] += 1
        self.sum_word_top[z] += 1
        self.Z[d][i] = z


    def theta_d(self, d):

        th_d = self.doc_top[d, :] / self.doc_top[d, :].sum()

        return th_d


    def theta_k(self, k):

        th_k = self.doc_top[:, k] / self.doc_top[:, k].sum()

        return th_k


    def phi_w(self, w):

        ph_w = self.top_word[:, w] / self.top_word[:, w].sum()

        return ph_w


    def phi_k(self, k):

        ph_k = self.top_word[k, :] / self.sum_word_top[k]

        return ph_k


    def _logp(self):
        """
        For testing
        """
        log_p = 0
        for d, doc in enumerate(self.W):
            for i, w in enumerate(doc):
                k = self.Z[d][i]
                log_p += log(self.phi_k(k)[w] * self.theta_d(d)[k])

        return log_p


    def logp(self):
        """
        """
        # This is slightly faster than distributing log over division
        log_kw = np.log(self.top_word / self.top_word.sum(1)[:, np.newaxis])
        log_dk = np.log(self.doc_top / self.doc_top.sum(1)[:, np.newaxis])

        log_p = 0
        for d, doc in enumerate(self.W):

            if len(doc) > 0:
                Z_d = self.Z[d]
                log_p += log_kw[Z_d, doc].sum()
                log_p += log_dk[d, :][Z_d].sum()

        return log_p


    @staticmethod
    def load(filename):

        from vsm.corpus import split_corpus
        from vsm.util.corpustools import empty_corpus

        print 'Loading LDA-Gibbs data from', filename
        arrays_in = np.load(filename)
        context_type = arrays_in['context_type'][()]
        K = arrays_in['K'][()]
        alpha = arrays_in['alpha'][()]
        beta = arrays_in['beta'][()]
        log_prob_init = arrays_in['log_prob_init'][()]

        m = LDAGibbs(empty_corpus(context_type),
                     context_type, K=K, alpha=alpha,
                     beta=beta, log_prob=log_prob_init)
        m.W = split_corpus(arrays_in['W_corpus'], arrays_in['W_indices'])
        m.V = arrays_in['V'][()]
        m.iterations = arrays_in['iterations'][()]
        if log_prob_init:
            m.log_prob = arrays_in['log_prob'].tolist()
        m.Z = split_corpus(arrays_in['Z_corpus'], arrays_in['Z_indices'])
        m.doc_top = arrays_in['doc_top']
        m.top_word = arrays_in['top_word']
        m.sum_word_top = arrays_in['sum_word_top']

        return m


    
    def save(self, filename):

        arrays_out = dict()
        arrays_out['W_corpus'] = np.array(np.hstack(self.W), dtype=np.int32)
        arrays_out['W_indices'] = np.cumsum([a.size for a in self.W])
        arrays_out['V'] = self.V
        arrays_out['iterations'] = self.iterations
        if hasattr(self, 'log_prob'):
            log_prob_init = True
            lp = np.array(self.log_prob,
                          dtype=[('i', np.int), ('v', np.float)])
            arrays_out['log_prob'] = lp
        else:
            log_prob_init = False
        arrays_out['Z_corpus'] = np.array(np.hstack(self.Z), dtype=np.int32)
        arrays_out['Z_indices'] = np.cumsum([a.size for a in self.Z])
        arrays_out['doc_top'] = self.doc_top
        arrays_out['top_word'] = self.top_word
        arrays_out['sum_word_top'] = self.sum_word_top
        arrays_out['context_type'] = self.context_type
        arrays_out['K'] = self.K
        arrays_out['alpha'] = self.alpha
        arrays_out['beta'] = self.beta
        arrays_out['log_prob_init'] = log_prob_init

        print 'Saving LDA-Gibbs model to', filename
        np.savez(filename, **arrays_out)
        

def test_LDAGibbs():

    from vsm.util.corpustools import random_corpus
    c = random_corpus(1000, 50, 6, 100)
    m = LDAGibbs(c, 'random', K=10)
    m.train(itr=20)

    return m



def test_dist_z():

    from vsm.util.corpustools import random_corpus
    c = random_corpus(500000, 10000, 0, 100)
    m = LDAGibbs(c, 'random', K=20)
    np.random.seed(0)
    m.train(itr=1)

    return m

    

def test_logp_fns():

    from vsm.util.corpustools import random_corpus
    c = random_corpus(500000, 10000, 0, 100)
    m = LDAGibbs(c, 'random', K=20)
    m.train(itr=1)
    
    logp_1 = m._logp()
    logp_2 = m.logp()

    assert np.allclose(logp_1, logp_2), (logp_1,logp_2)

    return m


def test_LDAGibbs_IO():

    from vsm.util.corpustools import random_corpus
    from tempfile import NamedTemporaryFile
    import os
    
    c = random_corpus(1000, 50, 6, 100)
    tmp = NamedTemporaryFile(delete=False, suffix='.npz')
    try:
        m0 = LDAGibbs(c, 'random', K=10)
        m0.train(itr=20)
        m0.save(tmp.name)
        m1 = LDAGibbs.load(tmp.name)
        assert m0.context_type == m1.context_type
        assert m0.K == m1.K
        assert m0.alpha == m1.alpha
        assert m0.beta == m1.beta
        assert m0.log_prob == m1.log_prob
        for i in xrange(max(len(m0.W), len(m1.W))):
            assert m0.W[i].all() == m1.W[i].all()
        assert m0.V == m1.V
        assert m0.iterations == m1.iterations
        for i in xrange(max(len(m0.Z), len(m1.Z))):
            assert m0.Z[i].all() == m1.Z[i].all()
        assert m0.doc_top.all() == m1.doc_top.all()
        assert m0.top_word.all() == m1.top_word.all()
        assert m0.sum_word_top.all() == m1.sum_word_top.all()
        m0 = LDAGibbs(c, 'random', K=10, log_prob=False)
        m0.train(itr=20)
        m0.save(tmp.name)
        m1 = LDAGibbs.load(tmp.name)
        assert not hasattr(m1, 'log_prob')
    finally:
        os.remove(tmp.name)
    
