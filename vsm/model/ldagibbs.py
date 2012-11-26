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



    # TODO: Verify that adding alpha and beta may be dispensed with

    def z_dist(self, d, w):

        top_inv = 1. / self.top

        dist = ((self.doc_top[d, :] + self.alpha) *
                (self.top_word[:, w] + self.beta) * top_inv)

        nc = 1. / dist.sum()

        return dist * nc



    def update_z(self, d, i, w):
        
        z = smpl_cat(self.z_dist(d, w))

        self.doc_top[d, z] += 1

        self.top_word[z, w] += 1

        self.top[z] += 1
        
        self.Z[d][i] = z




class LDAGibbs2(object):
    """
    K : the number of topics
    alpha : 
    beta : 
    """

    def theta(self, d):

        Z_d = self.Z[self.doc_ptr[d]: self.doc_ptr[d+1]]

        theta_d = np.bincount(Z_d, minlength = self.K) + self.alpha

        return theta_d



    def phi(self, w):

        i = self.W == w

        phi_w = np.bincount(self.Z[i], minlength = self.K) + self.beta

        return phi_w



    def n_Z(self):

        c = self.beta * self.V

        n = np.bincount(self.Z, minlength = self.K) + c

        return n




    def train(self, corpus, tok_name, 
              K=100, alpha = 0.01, beta = 0.01, itr=1000):

        #TODO: Support MaskedCorpus

        self.V = corpus.terms.shape[0]

        self.W = corpus.corpus

        self.doc_ptr = corpus._get_indices(tok_name)

        if not self.doc_ptr[-1] == len(self.W):

            self.doc_ptr = np.hstack([self.doc_ptr, [len(self.W)]])

        self.Z = np.zeros_like(self.W)

        self.K = K

        self.alpha = alpha
        
        self.beta = beta

        for t in xrange(itr):

            print 'Iteration', t

            for d in xrange(len(self.doc_ptr) - 1):

                for i in xrange(self.doc_ptr[d+1] - self.doc_ptr[d]):

                    w = self.W[self.doc_ptr[d] + i]

                    self.update_z(d, i, w)




    # TODO: Verify that adding alpha and beta may be dispensed with

    def z_dist(self, d, w):

        top_inv = 1. / self.n_Z()

        dist = ((self.theta(d) + self.alpha) *
                (self.phi(w) + self.beta) * top_inv)

        nc = 1. / dist.sum()

        return dist * nc


    def update_z(self, d, i, w):
        
        z = smpl_cat(self.z_dist(d, w))
        
        ind = self.doc_ptr[d] + i

        self.Z[ind] = z



def test_LDAGibbs():

    m = LDAGibbs()

    from vsm.corpus import random_corpus

    c = random_corpus(10000, 50, 6, 100)
    
    m.train(c, 'random', K=10, itr=100)

    return m



def test_LDAGibbs2():

    m = LDAGibbs2()

    from vsm.corpus import random_corpus

    c = random_corpus(10000, 50, 6, 100)
    
    m.train(c, 'random', K=10, itr=100)

    return m
