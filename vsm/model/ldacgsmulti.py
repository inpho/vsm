from sys import stdout
import multiprocessing as mp
import numpy as np
from ldafunctions import *


__all__ = [ 'LdaCgsMulti' ]



class LdaCgsMulti(object):
    """
    """
    def __init__(self, corpus, context_type, K=20, alpha=[], beta=[]):
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
        self.K = K
        global _K
        _K = mp.Value('i', K, lock=False)

        global _corpus
        _corpus = mp.Array('i', len(corpus.corpus), lock=False)
        _corpus[:] = corpus.corpus

        self.V = corpus.words.size
        global _V
        _V = mp.Value('i', self.V, lock=False)

        self.contexts = corpus.view_contexts(context_type, as_slices=True)
        self.context_type = context_type

        priors = init_priors_(self.V, self.K, beta, alpha)
        self.beta, self.alpha = priors

        # Topic posterior stored in shared array, initialized to zero
        LdaCgsMulti._init_word_top((np.zeros((_V.value, _K.value),
                                             dtype=np.float64)
                                    + self.beta).reshape(-1,))
        
        # Topic norms stored in a shared array, initialized to the
        # sums over the topic priors
        LdaCgsMulti._init_top_norms(1. / (np.ones(_K.value, dtype=np.float64)
                                          * self.beta.sum()))

        self.iteration = 0

        # The 0-th iteration is an initialization step, not a training step
        global _train
        _train = mp.Value('b', 0, lock=False)

        # Store log probability computations
        self.log_probs = []


    @staticmethod
    def _init_word_top(a):
        global _word_top
        _word_top = mp.Array('d', _V.value*_K.value, lock=False)
        _word_top[:] = a

    @staticmethod
    def _init_top_norms(a):
        global _top_norms
        _top_norms = mp.Array('d', _K.value, lock=False)
        _top_norms[:] = a


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

        :notes:
            Training sessions can be continued only if the previous
            training session of completed.
        """
        # Split contexts into an `n_proc`-length list of lists of
        # contexts
        if n_proc == 1:
            ctx_ls = [self.contexts]
        else:
            ctx_ls = np.array_split(self.contexts, n_proc-1)
            if len(ctx_ls) != n_proc:
                ctx_ls = np.array_split(self.contexts, n_proc)

        # Initialize arrays for storing Z and context posteriors for
        # each process
        if self.iteration == 0:
            self._Z = np.zeros(len(_corpus), dtype=np.int)
            self.top_doc = (np.zeros((_K.value, len(self.contexts)),
                                     dtype=np.float64)
                            + self.alpha)
        ctx_ls_flat = [slice(c[0].start, c[-1].stop) for c in ctx_ls]
        Z_ls = [self._Z[s] for s in ctx_ls_flat]
        ctx_sbls_spans = np.cumsum([len(ctx_sbls) for ctx_sbls in ctx_ls][:-1])
        top_doc_ls = np.split(self.top_doc, ctx_sbls_spans, axis=1)

        # Clean
        del self._Z, self.top_doc
        if hasattr(self, 'word_top'):
            del self.word_top

        p = mp.Pool(n_proc)

        itr += self.iteration
        while self.iteration < itr:
            if verbose:
                stdout.write('\rIteration %d: mapping  ' % self.iteration)
                stdout.flush()

            data = zip(ctx_ls, Z_ls, top_doc_ls)

            # For debugging
            # results = map(update, data)

            results = p.map(update, data)

            if verbose:
                stdout.write('\rIteration %d: reducing ' % self.iteration)
                stdout.flush()

            # Unzip results
            ctx_ls, Z_ls, top_doc_ls, word_top_ls, logp_ls = zip(*results)

            # Reduce word by topic matrices and store in global shared array
            word_top = (np.frombuffer(_word_top, dtype=np.float64)
                        + np.sum(word_top_ls, axis=0))
            top_norms = 1. / (word_top.reshape(_V.value, _K.value).sum(axis=0))
            _word_top[:] = word_top
            _top_norms[:] = top_norms
            del word_top, top_norms

            _train.value = 1

            lp = np.sum(logp_ls)
            self.log_probs.append((self.iteration, lp))

            if verbose:
                stdout.write('\rIteration %d: log_prob=' % self.iteration)
                stdout.flush()
                print '%f' % lp

            self.iteration += 1
                        
        p.close()
 
        # Final reduction includes assembling the Z and the context posteriors
        self._Z = np.hstack(Z_ls)
        self.top_doc = np.hstack(top_doc_ls)
        self.word_top = np.frombuffer(_word_top, dtype=np.float64)
        self.word_top = self.word_top.reshape(_V.value,_K.value)


    @property
    def Z(self):
        # For viewer until it gets updated
        return [self._Z[ctx] for ctx in self.contexts]


    @staticmethod
    def load(filename):
        """
        A static method for loading a saved LdaCgsMulti model.

        :param filename: Name of a saved model to be loaded.
        :type filename: string

        :returns: m : LdaCgsMulti object

        :See Also: :class:`numpy.load`
        """
        from vsm.corpus import BaseCorpus

        print 'Loading LdaCgsMulti data from', filename
        arrays_in = np.load(filename)
        context_type = arrays_in['context_type'][()]
        K = arrays_in['K'][()]
        
        try:
            alpha = arrays_in['alpha']
            beta = arrays_in['beta']
        except KeyError:
            # Compatibility
            alpha = arrays_in['doc_prior']
            beta = arrays_in['top_prior']

        c = BaseCorpus(arrays_in['corpus'],
                       context_types=[context_type],
                       context_data=[np.array([], dtype=[('idx', np.int)])],
                       remove_empty=False)
        m = LdaCgsMulti(c, context_type, K=K,
                        alpha=alpha, beta=beta)
        m.contexts = arrays_in['contexts']
        m.iteration = arrays_in['iteration'][()]
        m.log_probs = arrays_in['log_prob'].tolist()
        m._Z = arrays_in['Z']

        try:
            m.top_doc = arrays_in['top_doc']
        except KeyError:
            # Compatibility
            m.top_doc = arrays_in['top_ctx']

        m.word_top = arrays_in['word_top']

        LdaCgsMulti._init_word_top(m.word_top.reshape(-1,))
        LdaCgsMulti._init_top_norms(arrays_in['top_norms'])
        
        return m

    
    def save(self, filename):
        """
        Saves the model in `.npz` file.

        :param filename: Name of file to be saved.
        :type filename: string

        :See Also: :class:`numpy.savez`
        """
        arrays_out = dict()
        arrays_out['corpus'] = np.frombuffer(_corpus, np.int32)
        arrays_out['iteration'] = self.iteration
        dt = dtype=[('i', np.int), ('v', np.float)]
        arrays_out['log_prob'] = np.array(self.log_probs, dtype=dt)
        arrays_out['Z'] = self._Z
        arrays_out['top_doc'] = self.top_doc
        arrays_out['word_top'] = self.word_top
        arrays_out['context_type'] = self.context_type
        arrays_out['contexts'] = np.array(self.contexts)
        arrays_out['K'] = _K.value
        arrays_out['V'] = _V.value
        arrays_out['alpha'] = self.alpha
        arrays_out['beta'] = self.beta
        arrays_out['top_norms'] = np.frombuffer(_top_norms, np.float64)

        print 'Saving LdaCgsMulti model to', filename
        np.savez(filename, **arrays_out)

        

def update((ctx_sbls, Z, top_doc)):
    """
    For LdaCgsMulti
    """
    np.random.seed()    

    gbl_word_top = np.frombuffer(_word_top, dtype=np.float64)
    gbl_word_top = gbl_word_top.reshape(_V.value, _K.value)
    loc_word_top = gbl_word_top.copy()
    top_norms = np.frombuffer(_top_norms, dtype=np.float64).copy()

    log_p = 0
    log_wk = np.log(gbl_word_top * top_norms[np.newaxis, :])
    log_kc = np.log(top_doc / top_doc.sum(0)[np.newaxis, :])

    for i in xrange(len(ctx_sbls)):
        c = _corpus[ctx_sbls[i]]
        offset = ctx_sbls[i].start - ctx_sbls[0].start
        for j in xrange(len(c)):
            w,k = c[j],Z[offset+j]

            log_p += log_wk[w, k] + log_kc[k, i]

            if _train.value:
                loc_word_top[w, k] -= 1
                top_norms[k] *= 1. / (1 - top_norms[k])
                top_doc[k, i] -= 1

            dist = top_norms * loc_word_top[w,:] * top_doc[:,i]
            dist_cum = np.cumsum(dist)
            r = np.random.random() * dist_cum[-1]
            k = np.searchsorted(dist_cum, r)

            loc_word_top[w, k] += 1
            top_norms[k] *= 1. / (1 + top_norms[k]) 
            top_doc[k, i] += 1
            Z[offset+j] = k

    loc_word_top -= gbl_word_top

    return (ctx_sbls, Z, top_doc, loc_word_top.reshape(-1,), log_p)



#################################################################
#                            Tests
#################################################################

def test_LdaCgsMulti():

    from vsm.ext.corpusbuilders import random_corpus
    c = random_corpus(100, 5, 4, 20)
    m = LdaCgsMulti(c, 'document', K=3)
    m.train(itr=5, n_proc=2)

    return m


def test_LdaCgsMulti_IO():

    from vsm.ext.corpusbuilders import random_corpus
    from tempfile import NamedTemporaryFile
    import os
    
    c = random_corpus(1000, 50, 6, 100)
    tmp = NamedTemporaryFile(delete=False, suffix='.npz')
    try:
        m0 = LdaCgsMulti(c, 'document', K=10)
        m0.train(itr=20)
        c0 = np.frombuffer(_corpus, np.int32).copy()
        K0 = _K.value
        V0 = _V.value
        word_top0 = np.frombuffer(_word_top, np.float64).copy()
        top_norms0 = np.frombuffer(_top_norms, np.float64).copy()
        m0.save(tmp.name)
        m1 = LdaCgsMulti.load(tmp.name)
        c1 = np.frombuffer(_corpus, np.int32).copy()
        K1 = _K.value
        V1 = _V.value
        word_top1 = np.frombuffer(_word_top, np.float64).copy()
        top_norms1 = np.frombuffer(_top_norms, np.float64).copy()
        assert m0.context_type == m1.context_type
        assert (m0.alpha == m1.alpha).all()
        assert (m0.beta == m1.beta).all()
        assert m0.log_probs == m1.log_probs
        # for i in xrange(max(len(m0.W), len(m1.W))):
        #     assert m0.W[i].all() == m1.W[i].all()
        assert m0.iteration == m1.iteration
        assert (m0._Z == m1._Z).all()
        assert m0.top_doc.all() == m1.top_doc.all()
        assert m0.word_top.all() == m1.word_top.all()
        assert (c0==c1).all()
        assert K0==K1
        assert V0==V1
        assert (word_top0==word_top1).all()
        assert (top_norms0==top_norms1).all(), (top_norms0, top_norms1)
    finally:
        os.remove(tmp.name)
    

def test_continuation():
    """
    :note:
    Disable reseeding in `update` before running this test and use
    sequential mapping
    """
    from vsm.util.corpustools import random_corpus
    c = random_corpus(100, 5, 4, 20)
    
    m0 = LdaCgsMulti(c, 'random', K=3)
    np.random.seed(0)
    m0.train(itr=5, n_proc=2)
    m0.train(itr=5, n_proc=2)

    m1 = LdaCgsMulti(c, 'random', K=3)
    np.random.seed(0)
    m1.train(itr=10, n_proc=2)

    assert (m0.word_top==m1.word_top).all()
    assert (m0._Z==m1._Z).all()
    assert (m0.top_doc==m1.top_doc).all()
    assert m0.log_probs == m1.log_probs
