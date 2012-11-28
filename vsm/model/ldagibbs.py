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
    def theta(self, d, t):

        theta_dt = self.doc_top[d, t] / self.doc_top[:, t].sum()

        return theta_dt



    def phi(self, t, w):

        phi_tw = self.top_word[t, w] / self.top

        return phi_tw



    def logp(self):

        

    def train(self, corpus, tok_name, 
              K=100, alpha = 0.01, beta = 0.01, itr=1000):

        #TODO: Support MaskedCorpus

        V = corpus.terms.shape[0]

        W = corpus.view_tokens(tok_name)

        self.Z = [np.zeros_like(d) for d in W]

        self.alpha = alpha
        
        self.beta = beta

        self.doc_top = np.zeros((len(W), K)) + alpha

        self.top_word = np.zeros((K, V)) + beta

        self.top = (V * beta) + np.zeros(K)

        # Initialize

        for d, doc in enumerate(W):

            for i, w in enumerate(doc):

                self.update_z(d, i, w)

        # Iterate

        for t in xrange(itr):

            print 'Iteration', t

            for d, doc in enumerate(W):

                for i, w in enumerate(doc):
                    
                    z = self.Z[d][i]

                    self.doc_top[d, z] -= 1

                    self.top_word[z, w] -= 1

                    self.top[z] -= 1

                    self.update_z(d, i, w)




    def z_dist(self, d, w):

        top_inv = 1. / self.top

        dist = (self.doc_top[d, :] *
                self.top_word[:, w] * top_inv)

        nc = 1. / dist.sum()

        return dist * nc



    def update_z(self, d, i, w):
        
        z = smpl_cat(self.z_dist(d, w))

        self.doc_top[d, z] += 1

        self.top_word[z, w] += 1

        self.top[z] += 1
        
        self.Z[d][i] = z




def test_LDAGibbs():

    m = LDAGibbs()

    from vsm.corpus import random_corpus

    c = random_corpus(10000, 50, 6, 100)
    
    m.train(c, 'random', K=10, itr=100)

    return m
