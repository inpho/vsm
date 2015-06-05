from sys import stdout
import multiprocessing as mp
import numpy as np
from vsm.split import split_documents
from ldafunctions import load_lda
from ldacgsseq import *
from _cgs_update import cgs_update

import platform # For Windows comaptability

from progressbar import ProgressBar, Percentage, Bar


__all__ = [ 'LdaCgsMulti' ]



class LdaCgsMulti(LdaCgsSeq):
    """
    An implementation of LDA using collapsed Gibbs sampling with multi-processing.

    On Windows platforms, LdaCgsMulti is not supported. A NotImplementedError 
    will be raised notifying the user to use the LdaCgsSeq package. Users
    desiring a platform-independent fallback should use LDA(multiprocess=True) to
    initialize the object, which will return either a LdaCgsMulti or a LdaCgsSeq
    instance, depending on the platform, while raising a RuntimeWarning.
    """
    def __init__(self, corpus=None, context_type=None, K=20, V=0, 
                 alpha=[], beta=[], n_proc=2, seeds=None):
        """
        Initialize LdaCgsMulti.

        :param corpus: Source of observed data.
        :type corpus: `Corpus`
    
        :param context_type: Name of tokenization stored in `corpus` whose tokens
            will be treated as documents.
        :type context_type: string, optional

        :param K: Number of topics. Default is `20`.
        :type K: int, optional
    
        :param alpha: Context priors. Default is a flat prior of 0.01 
            for all contexts.
        :type alpha: list, optional
    
        :param beta: Topic priors. Default is 0.01 for all topics.
        :type beta: list, optional

        :param n_proc: Number of processors used for training. Default is
            2.
        :type n_proc: int, optional
        
        :param seeds: List of random seeds, one for each thread.
            The length of the list should be same as `n_proc`. Default is `None`.
        :type seeds: list of integers, optional
        """
	if platform.system() == 'Windows':
            raise NotImplementedError("""LdaCgsMulti is not implemented on 
            Windows. Please use LdaCgsSeq.""")

        self._read_globals = False
        self._write_globals = False

        self.n_proc = n_proc

        # set random seeds if unspecified
        if seeds is None:
            maxint = np.iinfo(np.int32).max
            seeds = [np.random.randint(0, maxint) for n in range(n_proc)]

        # check number of seeds == n_proc
        if len(seeds) != n_proc:
            raise ValueError("Number of seeds must equal number of processors " +
                             str(n_proc))

        # initialize random states
        self.seeds = seeds
        self._mtrand_states = [np.random.RandomState(seed).get_state() for seed in self.seeds]

        super(LdaCgsMulti, self).__init__(corpus=corpus, context_type=context_type,
                                          K=K, V=V, alpha=alpha, beta=beta)

        # delete LdaCgsSeq seed and state
        del self.seed
        del self._mtrand_state
        
        
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
            if not '_K' in globals():
                _K = mp.Value('i')
            _K.value = K
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
            if not '_V' in globals():
                _V = mp.Value('i')
            _V.value = V
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
            if not '_iteration' in globals():
                _iteration = mp.Value('i')
            _iteration.value = iteration
        else:
            self._iteration_local = iteration


    def train(self, n_iterations=500, verbose=1):
        """
        Takes an optional argument, `n_iterations` and updates the model
        `n_iterations` times.

        :param n_iterations: Number of iterations. Default is 500.
        :type n_iterations: int, optional

        :param verbose: If `True`, current number of iterations
            are printed out to notify the user. Default is `True`.
        :type verbose: boolean, optional
        """
        if mp.cpu_count() < self.n_proc:
            raise RuntimeError("Model seeded with more cores than available." +
                               " Requires {0} cores.".format(self.n_proc))

        self._move_locals_to_globals()

        docs = split_documents(self.corpus, self.indices, self.n_proc)

        doc_indices = [(0, len(docs[0]))]
        for i in xrange(len(docs)-1):
            doc_indices.append((doc_indices[i][1],
                                doc_indices[i][1] + len(docs[i+1])))

        p = mp.Pool(self.n_proc)

	if verbose == 1:
            pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=n_iterations).start()
        
        n_iterations += self.iteration
        iteration = 0

        while self.iteration < n_iterations:
            if verbose == 2:
                stdout.write('\rIteration %d: mapping  ' % self.iteration)
                stdout.flush()
        
            data = zip(docs, doc_indices, self._mtrand_states)

            # For debugging
	    # results = map(update, data)
	    if platform.system() == 'Windows':
                raise NotImplementedError("""LdaCgsMulti is not implemented on Windows. 
                Please use LdaCgsSeq.""")
            else:
                results = p.map(update, data)

            if verbose == 2:
                stdout.write('\rIteration %d: reducing ' % self.iteration)
                stdout.flush()
            
	    if verbose == 1:
                #print("Self iteration", self.iteration)
                pbar.update(iteration)

            (Z_ls, top_doc_ls, word_top_ls, logp_ls, mtrand_str_ls, 
             mtrand_keys_ls, mtrand_pos_ls, mtrand_has_gauss_ls, 
             mtrand_cached_gaussian_ls) = zip(*results)
            
            self._mtrand_states = zip(mtrand_str_ls, mtrand_keys_ls, mtrand_pos_ls, 
                                mtrand_has_gauss_ls, mtrand_cached_gaussian_ls)

            for t in xrange(len(results)):
                start, stop = docs[t][0][0], docs[t][-1][1]
                self.Z[start:stop] = Z_ls[t]
                self.top_doc[:, doc_indices[t][0]:doc_indices[t][1]] = top_doc_ls[t]
            self.word_top = self.word_top + np.sum(word_top_ls, axis=0)
            self.inv_top_sums = 1. / self.word_top.sum(0)
            lp = np.sum(logp_ls)
            self.log_probs.append((self.iteration, lp))

            if verbose == 2:
                stdout.write('\rIteration %d: log_prob=' % self.iteration)
                stdout.flush()
                print '%f' % lp

            iteration += 1
            self.iteration += 1

        if verbose == 1:
            pbar.finish()

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
	if platform.system() == 'Windows':
            raise NotImplementedError("""LdaCgsMulti is not implemented on 
            Windows. Please use LdaCgsSeq.""")
        return load_lda(filename, LdaCgsMulti)



def update((docs, doc_indices, mtrand_state)):
    """
    For LdaCgsMulti
    """
    start, stop = docs[0][0], docs[-1][1]

    corpus = np.frombuffer(_corpus, dtype='i')[start:stop]
    Z = np.frombuffer(_Z, dtype='i')[start:stop].copy()

    gbl_word_top = np.frombuffer(_word_top, dtype='d')
    gbl_word_top = gbl_word_top.reshape(_V.value, _K.value)
    loc_word_top = gbl_word_top.copy()
    inv_top_sums = np.frombuffer(_inv_top_sums, dtype='d').copy()

    top_doc = np.frombuffer(_top_doc, dtype='d')
    top_doc = top_doc.reshape(_K.value, top_doc.size/_K.value)
    top_doc = top_doc[:, doc_indices[0]:doc_indices[1]].copy()

    log_p = 0
    log_wk = np.log(gbl_word_top * inv_top_sums[np.newaxis, :])
    log_kc = np.log(top_doc / top_doc.sum(0)[np.newaxis, :])

    indices = np.array([(j - start) for (i,j) in docs], dtype='i')

    results = cgs_update(_iteration.value,
                         corpus,
                         loc_word_top,
                         inv_top_sums,
                         top_doc,
                         Z,
                         indices,
                         mtrand_state[0],
                         mtrand_state[1],
                         mtrand_state[2],
                         mtrand_state[3],
                         mtrand_state[4])

    (loc_word_top, inv_top_sums, top_doc, Z, log_p, mtrand_str, mtrand_keys, 
     mtrand_pos, mtrand_has_gauss, mtrand_cached_gaussian) = results

    loc_word_top -= gbl_word_top

    return (Z, top_doc, loc_word_top, log_p, 
            mtrand_str, mtrand_keys, mtrand_pos, 
            mtrand_has_gauss, mtrand_cached_gaussian)



#################################################################
#                            Demos
#################################################################


def demo_LdaCgsMulti(doc_len=500, V=100000, n_docs=100,
                     K=20, n_iterations=5, n_proc=2, 
                     corpus_seed=None, model_seeds=None):

    from vsm.extensions.corpusbuilders import random_corpus
    
    print 'Words per document:', doc_len
    print 'Words in vocabulary:', V
    print 'Documents in corpus:', n_docs
    print 'Number of topics:', K
    print 'Iterations:', n_iterations
    print 'Number of processors:', n_proc

    c = random_corpus(n_docs*doc_len, V, doc_len, doc_len+1, seed=corpus_seed)
    m = LdaCgsMulti(c, 'document', K=K, n_proc=n_proc, seeds=model_seeds)
    m.train(n_iterations=n_iterations, verbose=2)

    return m

