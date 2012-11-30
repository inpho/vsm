import numpy as np


# TODO: This could be made faster.

def smpl_cat(d):

    return np.random.multinomial(1, d).argmax()



class LDAGibbs(object):
    """
    K : the number of topics
    alpha : 
    beta : 
    """
    def __init__(self):

        self.iterations = 0

        self.log_probs = []

        

    def train(self, corpus=None, tok_name=None, 
              K=100, alpha = 0.01, beta = 0.01, 
              itr=1000, log_prob=True):

        #TODO: Support MaskedCorpus

        if not corpus == None:

            self.V = corpus.terms.shape[0]
            
            self.W = corpus.view_tokens(tok_name)

        self.Z = [np.zeros_like(d) for d in self.W]

        self.alpha = alpha
        
        self.beta = beta

        self.doc_top = np.zeros((len(self.W), K)) + alpha

        self.sum_doc_top = (K * alpha) + len(self.W)

        self.top_word = np.zeros((K, self.V)) + beta

        self.sum_word_top = (self.V * beta) + np.zeros(K)

        # Initialize

        for d, doc in enumerate(self.W):

            for i, w in enumerate(doc):

                self.update_z(d, i, w)

        # Iterate

        for t in xrange(self.iterations, self.iterations + itr):

            print 'Iteration', t

            self.iterations += 1
            
            if log_prob:

                self.log_probs.append((t, self.logp()))

            for d, doc in enumerate(self.W):

                for i, w in enumerate(doc):
                    
                    z = self.Z[d][i]

                    self.doc_top[d, z] -= 1

                    self.top_word[z, w] -= 1

                    self.sum_word_top[z] -= 1

                    self.update_z(d, i, w)




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
