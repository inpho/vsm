from sys import stdout
import multiprocessing as mp
import numpy as np
from vsm.split import split_documents
from ldafunctions import *


__all__ = [ 'LdaCgsMulti' ]



class LdaCgsMulti(object):
    """
    """
    def __init__(self, corpus=None, context_type=None, K=20, V=0, alpha=[], beta=[]):
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

        self.context_type = context_type
        self.K = K
        
        if corpus:
            self.V = corpus.words.size
            self.indices = corpus.view_contexts(context_type, as_indices=True)
            self.corpus = corpus.corpus
        else:
            self.V = V
            self.indices = []
            self.corpus = []

        self.Z = np.zeros_like(self.corpus, dtype=np.int)

        priors = init_priors(self.V, self.K, beta, alpha)
        self.beta, self.alpha = priors

        self.word_top = np.zeros((self.V, self.K), dtype=np.float64) + self.beta
        self.inv_top_sums = 1. / self.word_top.sum(0)
        self.top_doc = np.zeros((self.K, len(self.indices)),
                                dtype=np.float64) + self.alpha
        
        self.iteration = 0
        self.log_probs = []
        
        
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


    def train(self, itr=500, verbose=True, n_proc=2):
        """
        :param itr: Number of iterations. Default is 500.
        :type itr: int, optional

        :param verbose: If `True`, current number of iterations
            are printed out to notify the user. Default is `True`.
        :type verbose: boolean, optional

        :param n_proc: Number of processors used for training. Default is
            2.
        :type n_proc: int, optional
        """
        self._move_locals_to_globals()

        doc_partitions = split_documents(self.corpus, self.indices, n_proc)

        doc_indices = [(0, len(doc_partitions[0]))]
        for i in xrange(len(doc_partitions)-1):
            doc_indices.append((doc_indices[i][1],
                                doc_indices[i][1] + len(doc_partitions[i+1])))

        p = mp.Pool(n_proc)

        itr += self.iteration
        while self.iteration < itr:
            if verbose:
                stdout.write('\rIteration %d: mapping  ' % self.iteration)
                stdout.flush()
        
            data = zip(doc_partitions, doc_indices)

            # For debugging
            # results = map(update, data)

            results = p.map(update, data)

            if verbose:
                stdout.write('\rIteration %d: reducing ' % self.iteration)
                stdout.flush()

            Z_ls, top_doc_ls, word_top_ls, logp_ls = zip(*results)

            self.Z = np.hstack(Z_ls)
            self.word_top = self.word_top + np.sum(word_top_ls, axis=0)
            self.inv_top_sums = 1. / self.word_top.sum(0)
            self.top_doc = np.hstack(top_doc_ls)

            lp = np.sum(logp_ls)
            self.log_probs.append((self.iteration, lp))

            if verbose:
                stdout.write('\rIteration %d: log_prob=' % self.iteration)
                stdout.flush()
                print '%f' % lp

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

    
    def save(self, filename):
        """
        Saves the model in `.npz` file.

        :param filename: Name of file to be saved.
        :type filename: string

        :See Also: :class:`numpy.savez`
        """
        save_lda(self, filename)
        


def update((docs, doc_indices)):
    """
    For LdaCgsMulti
    """
    random_state = np.random.RandomState()

    start, stop = docs[0][0], docs[-1][1]
    corpus = _corpus[start:stop]
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

    for i in xrange(len(docs)):
        offset = docs[i][0] - docs[0][1]
        N = docs[i][1] - docs[i][0]
        for j in xrange(N):
            w, k = corpus[offset+j], Z[offset+j]

            log_p += log_wk[w, k] + log_kc[k, i]

            if _iteration.value > 0:
                loc_word_top[w, k] -= 1
                inv_top_sums[k] *= 1. / (1 - inv_top_sums[k])
                top_doc[k, i] -= 1

            dist = inv_top_sums * loc_word_top[w,:] * top_doc[:,i]
            k = categorical(dist, random_state=random_state)

            loc_word_top[w, k] += 1
            inv_top_sums[k] *= 1. / (1 + inv_top_sums[k]) 
            top_doc[k, i] += 1
            Z[offset+j] = k

    loc_word_top -= gbl_word_top

    return (Z, top_doc, loc_word_top, log_p)


#################################################################
#                            Demos
#################################################################


def demo_LdaCgsMulti(doc_len=500, V=100000, n_docs=100,
                     K=20, itr=5, n_proc=20):

    from vsm.extensions.corpusbuilders import random_corpus
    
    print 'Words per document:', doc_len
    print 'Words in vocabulary:', V
    print 'Documents in corpus:', n_docs
    print 'Number of topics:', K
    print 'Iterations:', itr
    print 'Number of processors:', n_proc

    c = random_corpus(n_docs*doc_len, V, doc_len, doc_len+1)
    m = LdaCgsMulti(c, 'document', K=K)
    m.train(itr=itr, n_proc=n_proc)

    return m


# def test_LdaCgsMulti_IO():

#     from vsm.extensions.corpusbuilders import random_corpus
#     from tempfile import NamedTemporaryFile
#     import os
    
#     c = random_corpus(1000, 50, 6, 100)
#     tmp = NamedTemporaryFile(delete=False, suffix='.npz')
#     try:
#         m0 = LdaCgsMulti(c, 'document', K=10)
#         m0.train(itr=20)
#         c0 = np.frombuffer(_corpus, np.int32).copy()
#         K0 = _K.value
#         V0 = _V.value
#         word_top0 = np.frombuffer(_word_top, np.float64).copy()
#         top_norms0 = np.frombuffer(_top_norms, np.float64).copy()
#         m0.save(tmp.name)
#         m1 = LdaCgsMulti.load(tmp.name)
#         c1 = np.frombuffer(_corpus, np.int32).copy()
#         K1 = _K.value
#         V1 = _V.value
#         word_top1 = np.frombuffer(_word_top, np.float64).copy()
#         top_norms1 = np.frombuffer(_top_norms, np.float64).copy()
#         assert m0.context_type == m1.context_type
#         assert (m0.alpha == m1.alpha).all()
#         assert (m0.beta == m1.beta).all()
#         assert m0.log_probs == m1.log_probs
#         # for i in xrange(max(len(m0.W), len(m1.W))):
#         #     assert m0.W[i].all() == m1.W[i].all()
#         assert m0.iteration == m1.iteration
#         assert (m0._Z == m1._Z).all()
#         assert m0.top_doc.all() == m1.top_doc.all()
#         assert m0.word_top.all() == m1.word_top.all()
#         assert (c0==c1).all()
#         assert K0==K1
#         assert V0==V1
#         assert (word_top0==word_top1).all()
#         assert (top_norms0==top_norms1).all(), (top_norms0, top_norms1)
#     finally:
#         os.remove(tmp.name)
        

# def test_continuation():
#     """
#     :note:
#     Disable reseeding in `update` before running this test and use
#     sequential mapping
#     """
#     from vsm.util.corpustools import random_corpus
#     c = random_corpus(100, 5, 4, 20)
    
#     m0 = LdaCgsMulti(c, 'random', K=3)
#     np.random.seed(0)
#     m0.train(itr=5, n_proc=2)
#     m0.train(itr=5, n_proc=2)

#     m1 = LdaCgsMulti(c, 'random', K=3)
#     np.random.seed(0)
#     m1.train(itr=10, n_proc=2)

#     assert (m0.word_top==m1.word_top).all()
#     assert (m0._Z==m1._Z).all()
#     assert (m0.top_doc==m1.top_doc).all()
#     assert m0.log_probs == m1.log_probs
