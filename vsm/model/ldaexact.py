import numpy as np
from itertools import product
from ldacgsseq import LdaCgsSeq


__all__ = [ 'LdaExact' ]



def uniquify(l):
    """Takes a list `l` and returns a list of the unique elements in `l`
    in the order in which they appeared.

    """
    mem = set([])
    out = []
    for e in l:
        if e not in mem:
            mem.add(e)
            out.append(e)
    return out


def productoid(A, n):

    prod = product(A, repeat=n)
    d = dict((i, A[:i]) for i in xrange(1, len(A)+1))

    for t in prod:
        elems = uniquify(t)
        if elems == d[len(elems)]:
            yield t


class LdaExact(LdaCgsSeq):


    @property
    def arg_maxima(self):
        if hasattr(self, '_arg_maxima'):
            return self._arg_maxima
        return []


    @arg_maxima.setter
    def arg_maxima(self, l):
        self._arg_maxima = l


    def _Z_values(self):
        
        A = range(self.K)
        p = productoid(A, len(self.corpus))
        for t in p:
            yield np.array(t, dtype=np.int)


    def _init_model(self, Z):
        m = LdaCgsSeq(context_type=self.context_type,
                      K=self.K, V=self.V, alpha=self.alpha, beta=self.beta)
        m.corpus = self.corpus
        m.V = self.V
        m.indices = self.indices
        m.Z = Z
        m._compute_top_doc()
        m._compute_word_top()
        m.inv_top_sums = 1. / self.word_top.sum(0)
        m.iteration = 1
        m.log_probs = [(1, m._compute_log_prob())]
        return m
        

    def _log_probs(self):

        Z = self._Z_values()

        for next_Z in Z:
            m = self._init_model(next_Z)
            yield (next_Z, m.log_probs[0][1])


    def arg_max(self, verbose=1):
        
        max_log_prob = -np.inf
        maxima = []

        log_probs = self._log_probs()
        for (Z, log_prob) in log_probs:
            if log_prob == max_log_prob:
                maxima.append((Z, log_prob))
            elif log_prob > max_log_prob:
                max_log_prob = log_prob
                maxima = [(Z, log_prob)]

        self.arg_maxima = maxima
        self.Z = maxima[0][0]
        self._compute_top_doc()
        self._compute_word_top()
        self.inv_top_sums = 1. / self.word_top.sum(0)
        self.iteration = 1
        self.log_probs = [(1, max_log_prob)]

        if verbose > 0:
            print 'Number of maxima:', len(self.arg_maxima)


    def all_estimates(self):
        
        for (Z, log_prob) in self.arg_maxima:
            yield self._init_model(Z)

