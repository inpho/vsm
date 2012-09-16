import numpy as np

from vsm import model



class BeagleEnvironment(model.Model):

    def train(self, corpus, n_columns=2048, dtype=np.float32):

        shape = (corpus.terms.shape[0], n_columns)

        self.matrix = dtype(np.random.random(shape))
        self.matrix *= 2
        self.matrix -= 1

        norms = np.sum(self.matrix**2, axis=1)**(.5)

        self.matrix /= norms[:, np.newaxis]
