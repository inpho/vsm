"""
Several solutions in this implementation were found in 

Nakatani Shuyo's `iirda` at https://github.com/shuyo/iir

and in 

Hanna Wallach's `python-lda` at
https://github.com/hannawallach/python-lda
"""
from sys import stdout

import numpy as np



def smpl_cat(d):
    """
    Takes an array of probabilities d and returns a sample from the
    categorical distribution parameterized by d.
    """
    return np.random.multinomial(1, d).argmax()



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
    tok_name : string
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
        records them in `log_probs`. Default is `True`.

    Attributes
    ----------
    W : list of integer arrays
        List of documents, which are extracted from the input Corpus object
    V : int
        Number of unique terms in the corpus
    Z : list of integer arrays
        Topic assignments for every term coordinate in the corpus
    iterations : int
        Number of past iterations of the update rule
    doc_top : 2-dim floating point array
        Stores the unnormalized estimated posterior distribution over
        topics for each document in a D x K matrix
    top_word : 2-dim floating point array
        Stores the unnormalized estimated posterior distribution over
        terms for each topic in a K x V matrix
    sum_doc_top : 1-dim floating point array
        Stores the sum of documents over topics
    sum_word_top : 1-dim floating point array
        Stores the sum of terms over topics

    Methods
    -------
    train
        Takes an optional argument `itr`, which defaults to 1000, and
        updates the model `itr` times.
    update_z
        Takes a document index `d`, a term index `i` relative to that
        document and a term `w` and updates the model.
    z_dist
        Takes a document index `d` and a term `w` and computes the
        distribution over topics for `w` in `d`
    phi_t
        Takes a topic index `t` and returns the estimated posterior
        distribution over terms for `t`
    phi_w
        Takes a term `w` and returns the estimated posterior
        distribution over topics for `w`
    theta_d
        Takes a document index `d` and returns the estimated posterior
        distribution over topics for `d`
    logp
        Compute the log probability of the corpus `W` given the
        estimated values of the latent variables `phi`, `theta` and
        `Z`

    """
    def __init__(self, corpus, tok_name,
                 K=100, alpha = 0.01, beta = 0.01, 
                 log_prob=True):

        #TODO: Support MaskedCorpus

        self.K = K

        self.alpha = alpha

        self.beta = beta
        
        self.W = corpus.view_tokens(tok_name)

        self.V = corpus.terms.shape[0]
            
        self.iterations = 0

        if log_prob:

            self.log_prob = []
        
        # Initialize

        self.Z = [np.zeros_like(d) for d in self.W]

        self.doc_top = np.zeros((len(self.W), K)) + alpha

        self.sum_doc_top = (K * alpha) + len(self.W)

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

        dist = (self.doc_top[d, :] *
                self.top_word[:, w] * sum_word_top_inv)

        nc = 1. / dist.sum()

        return dist * nc



    def update_z(self, d, i, w):
        
        z = smpl_cat(self.z_dist(d, w))

        self.doc_top[d, z] += 1

        self.top_word[z, w] += 1

        self.sum_word_top[z] += 1
        
        self.Z[d][i] = z



    def theta_d(self, d):

        th_d = self.doc_top[d, :] / self.sum_doc_top

        return th_d



    def phi_w(self, w):

        ph_w = self.top_word[:, w] / self.sum_word_top

        return ph_w



    def phi_t(self, t):

        ph_t = self.top_word[t, :] / self.sum_word_top[t]

        return ph_t



    def logp(self):

        log_p = 0

        for d, doc in enumerate(self.W):

            for i, w in enumerate(doc):

                log_p -= np.dot(self.theta_d(d), self.phi_w(w))

        return log_p

        

def test_LDAGibbs():

    m = LDAGibbs()

    from vsm.corpus import random_corpus

    c = random_corpus(10000, 50, 6, 100)
    
    m.train(c, 'random', K=10, itr=100)

    return m
