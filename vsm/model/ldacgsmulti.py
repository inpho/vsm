from sys import stdout
import multiprocessing as mp
import numpy as np

from ldagibbs import smpl_cat



class LdaCgsMulti(object):
    """
    """
    def __init__(self, corpus, context_type,
                 K=100, top_prior = [], ctx_prior = []):

        # The width of the word by topic matrix and the height of the
        # topic by context matrix
        global _K
        _K = mp.Value('i', K, lock=False)

        # Store corpus as shared array.
        global _corpus
        _corpus = mp.Array('i', len(corpus.corpus), lock=False)
        _corpus[:] = corpus.corpus

        # The height of the word by topic matrix
        global _m_words
        _m_words = mp.Value('i', corpus.words.size, lock=False)

        # Chunks of contexts are the primary data over which we'll map
        # the update rule
        self.contexts = corpus.view_contexts(context_type, as_slices=True)

        # Store context_type for later viewing
        self.context_type = context_type

        # Topic and context priors; set defaults if need be
        if len(top_prior) > 0:
            self.top_prior = np.array(top_prior, dtype=np.float64)
            self.top_prior = self.top_prior.reshape(_m_words.value,1)
        else:
            # Default is a flat prior of .01
            self.top_prior = np.ones((_m_words.value,1), dtype=np.float64) * .01

        if len(ctx_prior) > 0:
            self.ctx_prior = np.array(ctx_prior,
                                      dtype=np.float64).reshape(_K.value,1)
        else:
            # Default is a flat prior of .01
            self.ctx_prior = np.ones((_K.value,1), dtype=np.float64) * .01

        # Topic posterior stored in shared array, initialized to zero        
        global _word_top
        _word_top = mp.Array('d', _m_words.value*_K.value, lock=False)
        _word_top[:] = (np.zeros((_m_words.value, _K.value), np.float64)
                        + self.top_prior).reshape(-1,)

        # Topic norms stored in a shared array, initialized to the
        # sums over the topic priors
        global _top_norms
        _top_norms = mp.Array('d', _K.value, lock=False)
        _top_norms[:] = 1. / (np.ones(_K.value, dtype=np.float64)
                              * self.top_prior.sum())

        self.iterations = 0

        # The 0-th iteration is an initialization step, not a training step
        global _train
        _train = mp.Value('b', 0, lock=False)

        # Store log probability computations
        self.log_prob = []

        
    def train(self, itr=500, verbose=True, n_proc=2):

        #TODO: Enable training continuations

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
        spans = [(c[-1].stop - c[0].start) for c in ctx_ls]
        if self.iterations == 0:
            Z_ls = [np.zeros(n, dtype=np.int) for n in spans]
            top_ctx_ls = [(np.zeros((_K.value, len(c)), dtype=np.float64)
                           + self.ctx_prior) for c in ctx_ls]
        else:
            raise Exception('Training continuations not yet implemented.')

        p=mp.Pool(n_proc)

        for t in xrange(self.iterations, self.iterations + itr):        
            if verbose:
                stdout.write('\rIteration %d: mapping  ' % t)
                stdout.flush()

            data = zip(ctx_ls, Z_ls, top_ctx_ls)

            # For debugging
            # results = map(update, data)

            results = p.map(update, data)

            if verbose:
                stdout.write('\rIteration %d: reducing ' % t)
                stdout.flush()

            # Unzip results
            ctx_ls, Z_ls, top_ctx_ls, word_top_ls, logp_ls = zip(*results)

            # Reduce word by topic matrices and store in global shared array
            word_top = (np.frombuffer(_word_top, dtype=np.float64)
                        + np.sum(word_top_ls, axis=0))
            top_norms = 1. / (word_top.reshape(_m_words.value, _K.value).sum(axis=0))
            _word_top[:] = word_top
            _top_norms[:] = top_norms
            del word_top, top_norms

            _train.value = 1

            lp = np.sum(logp_ls)
            self.log_prob.append((t, lp))

            if verbose:
                stdout.write('\rIteration %d: log_prob=' % t)
                stdout.flush()
                print '%f' % lp

        p.close()
 
        # Final reduction includes assembling the Z and the context posteriors
        Z = np.hstack(Z_ls)
        top_ctx = np.hstack(top_ctx_ls)
        word_top = np.frombuffer(_word_top,
                                 dtype=np.float64).reshape(_m_words.value,_K.value)

        # TODO: For legacy viewer only. Update viewer and remove or update these.
        self.top_word = word_top.T
        self.doc_top = top_ctx.T
        self.W = [np.array(_corpus[ctx], dtype=np.int) for ctx in self.contexts]
        self.Z = [Z[ctx] for ctx in self.contexts]


    @staticmethod
    def load(filename):

        from vsm.util.corpustools import empty_corpus

        print 'Loading LdaCgsMulti data from', filename
        arrays_in = np.load(filename)
        context_type = arrays_in['context_type'][()]
        K = arrays_in['K'][()]
        ctx_prior = arrays_in['ctx_prior']
        top_prior = arrays_in['top_prior']
        e = empty_corpus(context_type)
        # hack
        e.words = np.empty(arrays_in['m_words'], np.int16)

        m = LdaCgsMulti(e, context_type, K=K, ctx_prior=ctx_prior,
                        top_prior=top_prior)
        m.contexts = arrays_in['contexts']
        m.W = [arrays_in['corpus'][ctx] for ctx in m.contexts]
        m.iterations = arrays_in['iterations'][()]
        m.log_prob = arrays_in['log_prob'].tolist()
        m.Z = [arrays_in['Z'][ctx] for ctx in m.contexts]
        m.doc_top = arrays_in['doc_top']
        m.top_word = arrays_in['top_word']

        return m

    
    def save(self, filename):

        arrays_out = dict()
        arrays_out['corpus'] = np.frombuffer(_corpus, np.int32)
        arrays_out['iterations'] = self.iterations
        dt = dtype=[('i', np.int), ('v', np.float)]
        arrays_out['log_prob'] = np.array(self.log_prob, dtype=dt)
        arrays_out['Z'] = np.array(np.hstack(self.Z), dtype=np.int32)
        arrays_out['doc_top'] = self.doc_top
        arrays_out['top_word'] = self.top_word
        arrays_out['context_type'] = self.context_type
        arrays_out['contexts'] = np.array(self.contexts)
        arrays_out['K'] = _K.value
        arrays_out['m_words'] = _m_words.value
        arrays_out['ctx_prior'] = self.ctx_prior
        arrays_out['top_prior'] = self.top_prior
        arrays_out['top_norms'] = np.frombuffer(_top_norms, np.float64)

        print 'Saving LdaCgsMulti model to', filename
        np.savez(filename, **arrays_out)

        

def update((ctx_sbls, Z, top_ctx)):
    """
    For LdaCgsMulti
    """
    np.random.seed()    

    gbl_word_top = np.frombuffer(_word_top, dtype=np.float64).copy()
    gbl_word_top = gbl_word_top.reshape(_m_words.value, _K.value)
    loc_word_top = gbl_word_top.copy()
    top_norms = np.frombuffer(_top_norms, dtype=np.float64).copy()

    log_p = 0
    log_wk = np.log(gbl_word_top * top_norms[np.newaxis, :])
    log_kc = np.log(top_ctx / top_ctx.sum(0)[np.newaxis, :])

    for i in xrange(len(ctx_sbls)):
        c = _corpus[ctx_sbls[i]]
        offset = ctx_sbls[i].start - ctx_sbls[0].start
        for j in xrange(len(c)):
            w,k = c[j],Z[offset+j]

            log_p += log_wk[w, k] + log_kc[k, i]

            if _train.value:
                loc_word_top[w, k] -= 1
                top_norms[k] *= 1. / (top_norms[k] - 1) 
                top_ctx[k, i] -= 1

            dist = top_norms * loc_word_top[w,:] * top_ctx[:,i]
            dist_cum = np.cumsum(dist)
            r = np.random.random() * dist_cum[-1]
            k = np.searchsorted(dist_cum, r)

            loc_word_top[w, k] += 1
            top_norms[k] *= 1. / (top_norms[k] + 1) 
            top_ctx[k, i] += 1
            Z[offset+j] = k

    loc_word_top -= gbl_word_top

    return (ctx_sbls, Z, top_ctx, loc_word_top.reshape(-1,), log_p)



#################################################################
#                            Tests
#################################################################

def test_LdaCgsMulti():

    from vsm.util.corpustools import random_corpus
    c = random_corpus(100, 5, 4, 20)
    m = LdaCgsMulti(c, 'random', K=3)
    m.train(itr=5, n_proc=1)

    return m


def test_LdaCgsMulti_IO():

    from vsm.util.corpustools import random_corpus
    from tempfile import NamedTemporaryFile
    import os
    
    c = random_corpus(1000, 50, 6, 100)
    tmp = NamedTemporaryFile(delete=False, suffix='.npz')
    try:
        m0 = LdaCgsMulti(c, 'random', K=10)
        m0.train(itr=20)
        m0.save(tmp.name)
        m1 = LdaCgsMulti.load(tmp.name)
        assert m0.context_type == m1.context_type
        assert (m0.ctx_prior == m1.ctx_prior).all()
        assert (m0.top_prior == m1.top_prior).all()
        assert m0.log_prob == m1.log_prob
        for i in xrange(max(len(m0.W), len(m1.W))):
            assert m0.W[i].all() == m1.W[i].all()
        assert m0.iterations == m1.iterations
        for i in xrange(max(len(m0.Z), len(m1.Z))):
            assert m0.Z[i].all() == m1.Z[i].all()
        assert m0.doc_top.all() == m1.doc_top.all()
        assert m0.top_word.all() == m1.top_word.all()
    finally:
        os.remove(tmp.name)
    
