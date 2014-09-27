import itertools
from sys import stdout
import multiprocessing as mp
import numpy as np
from vsm.split import split_documents
from ldafunctions import load_lda
from ldacgsseq import *

import pyximport; pyximport.install()
from _cgs_update import cgs_update


__all__ = [ 'LdaCgsMulti' ]



class LdaCgsMulti(LdaCgsSeq):
    """
    """
    def __init__(self, corpus=None, context_type=None, K=20, V=0, 
                 alpha=[], beta=[], seed=0):
        """
        Initialize LdaCgsMulti.

        :param corpus: Source of observed data.
        :type corpus: `Corpus`
    
        :param context_type: Name of tokenization stored in `corpus` whose tokens
            will be treated as documents.
        :type context_type: string, optional

        :param K: Number of topics. Default is `20`.
        :type K: int, optional
    
        :param beta: Topic priors. Default is 0.01 for all topics.
        :type beta: list, optional
    
        :param alpha: Context priors. Default is a flat prior of 0.01 
            for all contexts.
        :type alpha: list, optional
        """
        self._read_globals = False
        self._write_globals = False

        super(LdaCgsMulti, self).__init__(corpus=corpus, context_type=context_type,
                                          K=K, V=V, alpha=alpha, beta=beta,
                                          seed=seed)
        
        
    def _move_globals_to_locals(self):
        
        self._write_globals = False

        self.K = self.K
        self.V = self.V
        self.corpus = self.corpus
        self.Z = self.Z
        self.word_top = self.word_top
        self.inv_top_sums = self.inv_top_sums
        self.top_doc = self.top_doc
        self.iteration = self.iteration
        self.seed = self.seed

        self._read_globals = False

        global _K, _V, _corpus, _Z, _word_top, _inv_top_sums
        global _top_doc, _iteration
        del (_K, _V, _corpus, _Z, _word_top, _inv_top_sums,
             _top_doc, _iteration)


    def _move_locals_to_globals(self):
        
        self._write_globals = True
        
        self.K = self.K
        self.V = self.V
        self.corpus = self.corpus
        self.Z = self.Z
        self.word_top = self.word_top
        self.inv_top_sums = self.inv_top_sums
        self.top_doc = self.top_doc
        self.iteration = self.iteration
        self.seed = self.seed

        self._read_globals = True

        del (self._K_local, self._V_local, self._corpus_local, self._Z_local,
             self._word_top_local, self._inv_top_sums_local,
             self._top_doc_local, self._iteration_local)
        

    @property
    def word_top(self):
        if self._read_globals:
            return np.frombuffer(_word_top, np.float64).reshape(self.V, self.K)
        return self._word_top_local
        
    @word_top.setter
    def word_top(self, a):
        if self._write_globals:
            global _word_top
            if not '_word_top' in globals():
                _word_top = mp.Array('d', self.V * self.K, lock=False)
            _word_top[:] = a.reshape(-1,)
        else:
            self._word_top_local = a


    @property
    def inv_top_sums(self):
        if self._read_globals:
            return np.frombuffer(_inv_top_sums, np.float64)
        return self._inv_top_sums_local

    @inv_top_sums.setter
    def inv_top_sums(self, a):
        if self._write_globals:
            global _inv_top_sums
            if not '_inv_top_sums' in globals():
                _inv_top_sums = mp.Array('d', self.K, lock=False)
            _inv_top_sums[:] = a
        else:
            self._inv_top_sums_local = a


    @property
    def top_doc(self):
        if self._read_globals:
            top_doc = np.frombuffer(_top_doc, np.float64)
            return top_doc.reshape(self.K, len(self.indices))
        return self._top_doc_local
        
    @top_doc.setter
    def top_doc(self, a):
        if self._write_globals:
            global _top_doc
            if not '_top_doc' in globals():
                _top_doc = mp.Array('d', self.K * len(self.indices), lock=False)
            _top_doc[:] = a.reshape(-1,)
        else:
            self._top_doc_local = a


    @property
    def corpus(self):
        if self._read_globals:
            return np.frombuffer(_corpus, np.int32)
        return self._corpus_local

    @corpus.setter
    def corpus(self, a):
        if self._write_globals:
            global _corpus
            if not '_corpus' in globals():
                _corpus = mp.Array('i', len(a), lock=False)
            _corpus[:] = a
        else:
            self._corpus_local = a


    @property
    def Z(self):
        if self._read_globals:
            return np.frombuffer(_Z, np.int32)
        return self._Z_local

    @Z.setter
    def Z(self, a):
        if self._write_globals:
            global _Z
            if not '_Z' in globals():
                _Z = mp.Array('i', len(a), lock=False)
            _Z[:] = a
        else:
            self._Z_local = a


    @property
    def K(self):
        if self._read_globals:
            return _K.value
        return self._K_local

    @K.setter
    def K(self, K):
        if self._write_globals:
            global _K
            _K = mp.Value('i', K, lock=False)
        else:
            self._K_local = K


    @property
    def V(self):
        if self._read_globals:
            return _V.value
        return self._V_local

    @V.setter
    def V(self, V):
        if self._write_globals:
            global _V
            _V = mp.Value('i', V, lock=False)
        else:
            self._V_local = V


    @property
    def iteration(self):
        if self._read_globals:
            return _iteration.value
        return self._iteration_local

    @iteration.setter
    def iteration(self, iteration):
        if self._write_globals:
            global _iteration
            _iteration = mp.Value('i', iteration, lock=False)
        else:
            self._iteration_local = iteration


    def train(self, n_iterations=500, verbose=True, n_proc=2):
        """
        :param n_iterations: Number of iterations. Default is 500.
        :type n_iterations: int, optional

        :param verbose: If `True`, current number of iterations
            are printed out to notify the user. Default is `True`.
        :type verbose: boolean, optional

        :param n_proc: Number of processors used for training. Default is
            2.
        :type n_proc: int, optional
        """
        self._move_locals_to_globals()

        docs = split_documents(self.corpus, self.indices, n_proc)

        doc_indices = [(0, len(docs[0]))]
        for i in xrange(len(docs)-1):
            doc_indices.append((doc_indices[i][1],
                                doc_indices[i][1] + len(docs[i+1])))

        p = mp.Pool(n_proc)

        n_iterations += self.iteration
        while self.iteration < n_iterations:
            if verbose:
                stdout.write('\rIteration %d: mapping  ' % self.iteration)
                stdout.flush()
        
            data = zip(docs, doc_indices, itertools.repeat(self.seed))

            # For debugging
            # results = map(update, data)

            results = p.map(update, data)

            if verbose:
                stdout.write('\rIteration %d: reducing ' % self.iteration)
                stdout.flush()

            Z_ls, top_doc_ls, word_top_ls, logp_ls = zip(*results)
            
            for t in xrange(len(results)):
                start, stop = docs[t][0][0], docs[t][-1][1]
                self.Z[start:stop] = Z_ls[t]
                self.top_doc[:, doc_indices[t][0]:doc_indices[t][1]] = top_doc_ls[t]
            self.word_top = self.word_top + np.sum(word_top_ls, axis=0)
            self.inv_top_sums = 1. / self.word_top.sum(0)
            lp = np.sum(logp_ls)
            self.log_probs.append((self.iteration, lp))

            if verbose:
                stdout.write('\rIteration %d: log_prob=' % self.iteration)
                stdout.flush()
                print '%f (seed: %s)' % (lp, self.seed)

            self.iteration += 1

        p.close()
        self._move_globals_to_locals()


    @staticmethod
    def load(filename):
        """
        A static method for loading a saved LdaCgsMulti model.

        :param filename: Name of a saved model to be loaded.
        :type filename: string

        :returns: m : LdaCgsMulti object

        :See Also: :class:`numpy.load`
        """
        return load_lda(filename, LdaCgsMulti)



def update((docs, doc_indices, seed)):
    """
    For LdaCgsMulti
    """
    start, stop = docs[0][0], docs[-1][1]

    corpus = np.frombuffer(_corpus, dtype=np.int32)[start:stop]
    Z = np.frombuffer(_Z, dtype=np.int32)[start:stop].copy()

    gbl_word_top = np.frombuffer(_word_top, dtype=np.float64)
    gbl_word_top = gbl_word_top.reshape(_V.value, _K.value)
    loc_word_top = gbl_word_top.copy()
    inv_top_sums = np.frombuffer(_inv_top_sums, dtype=np.float64).copy()

    top_doc = np.frombuffer(_top_doc, dtype=np.float64)
    top_doc = top_doc.reshape(_K.value, top_doc.size/_K.value)
    top_doc = top_doc[:, doc_indices[0]:doc_indices[1]].copy()

    log_p = 0
    log_wk = np.log(gbl_word_top * inv_top_sums[np.newaxis, :])
    log_kc = np.log(top_doc / top_doc.sum(0)[np.newaxis, :])

    indices = np.array([(j - start) for (i,j) in docs], dtype='l')
    Z = np.array(Z, dtype='l')

    results = cgs_update(_iteration.value,
                         corpus,
                         loc_word_top,
                         inv_top_sums,
                         top_doc,
                         Z,
                         indices,
                         seed)

    loc_word_top, inv_top_sums, top_doc, Z, log_p = results

    loc_word_top -= gbl_word_top

    return (Z, top_doc, loc_word_top, log_p)



#################################################################
#                            Demos
#################################################################


def demo_LdaCgsMulti(doc_len=500, V=100000, n_docs=100,
                     K=20, n_iterations=5, n_proc=20):

    from vsm.extensions.corpusbuilders import random_corpus
    
    print 'Words per document:', doc_len
    print 'Words in vocabulary:', V
    print 'Documents in corpus:', n_docs
    print 'Number of topics:', K
    print 'Iterations:', n_iterations
    print 'Number of processors:', n_proc

    c = random_corpus(n_docs*doc_len, V, doc_len, doc_len+1)
    m = LdaCgsMulti(c, 'document', K=K)
    m.train(n_iterations=n_iterations, n_proc=n_proc)

    return m
