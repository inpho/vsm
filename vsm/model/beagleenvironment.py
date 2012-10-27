import numpy as np

from vsm import model



class BeagleEnvironment(model.Model):

    def train(self, corpus, n_columns=2048, dtype=np.float32, rand_pool=None):


        if rand_pool:

            shape = (corpus.terms.shape[0], n_columns)
                        
            self.matrix = np.empty(shape, dtype=dtype)
            
            for i in xrange(shape[0]):

                v = rand_pool.request_vector(corpus.terms[i])

                self.matrix[i, :] = v[:]
                

        else:

            shape = (corpus.terms.shape[0], n_columns)

            self.matrix = np.asarray(np.random.random(shape), dtype=dtype)

            self.matrix *= 2

            self.matrix -= 1
            
            norms = np.sum(self.matrix**2, axis=1)**(.5)
            
            self.matrix /= norms[:, np.newaxis]



class RandomVectorPool(object):
    """
    """
    def __init__(self, seed=None, n_columns=2048, dtype=np.float32):

        self.n_columns = n_columns

        self.dtype = dtype

        self.rstate = np.random.mtrand.RandomState()

        self.rstate.seed(seed)

        self.terms = []

        self.matrix = np.empty(0)



    def request_vector(self, term):

        if term in self.terms:

            return self.matrix[self.terms.index(term)]

        else:

            v = np.asarray(self.rstate.rand(self.n_columns),
                           dtype=self.dtype)

            v = v * 2 - 1

            v = v / np.dot(v, v)**.5

            if self.matrix.any():
                
                self.matrix = np.vstack([self.matrix, v])

            else:

                self.matrix = v[np.newaxis, :]

            self.terms += [term]

            return v



    def save(self, filename):

        with open(filename, 'w') as f:

            pickle.dump(self)
