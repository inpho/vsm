import os
import shutil
import tempfile
import multiprocessing as mp

import numpy as np
from numpy import dual

from vsm import rand_pt_unit_sphere, naive_cconv
from vsm import corpus as corp
from vsm import model
from vsm.model import beagleenvironment as be



def two_rand_perm(n, seed=None):

    np.random.seed(seed)
    
    perm1 = np.random.permutation(n)

    while True:

        perm2 = np.random.permutation(n)

        if not (perm2 == perm1).all():

            break
        
    return perm1, perm2



def mk_b_conv(n, rand_perm=None):

    if rand_perm is None:

        rand_perm = two_rand_perm(n)
    
    def b_conv(v1, v2):
        
        w1 = dual.fft(v1[rand_perm[0]])

        w2 = dual.fft(v2[rand_perm[1]])

        return np.real_if_close(dual.ifft(w1 * w2))

    return b_conv



def ngram_slices(i, n, l):
    """
    Given index i, n-gram width n and array length l, returns slices
    for all n-grams containing an ith element
    """
    out = []

    a = i - n + 1

    if a < 0:

        a = 0

    b = i + 1

    if b + n > l:

        b = l - n + 1

    d = b - a

    for k in xrange(d):

        start = a + k

        stop = start + n
            
        out.append(slice(start, stop))

    return out



def reduce_ngrams(fn, a, n, i, flat=True):
    """
    Given array a, reduce with fn all n-grams with width n and less and
    that contain the element at i.

    Memoizes.
    """
    m = n if n < a.shape[0] else a.shape[0]

    out = { 1: { i: a[i] } }
    
    for j in xrange(2, m + 1):

        slices = ngram_slices(i, j, a.shape[0])

        init = slices[0]

        out[j] = { init.start: reduce(fn, a[init]) }

        for s in slices[1:]:

            prev = out[j - 1][s.start]

            out[j][s.start] = fn(prev, a[s.stop - 1])

    # Delete the 1-gram

    del out[1]

    if flat:

        out = [v for d in out.values() for v in d.values()]
    
    return out



class BeagleOrderSingle(model.Model):

    def train(self,
              corpus,
              env_matrix=None,
              psi=None,
              rand_perm=None,
              n_columns=2048,
              lmda = 7,
              tok_name='sentences'):
        
        if env_matrix is None:
            
            m = be.BeagleEnvironment()

            m.train(corpus, n_columns=n_columns)

            env_matrix = m.matrix[:, :]

        b_conv = mk_b_conv(n_columns, rand_perm)

        if psi is None:

            psi = rand_pt_unit_sphere(env_matrix.shape[1])

        self.matrix = np.zeros_like(env_matrix)



        if isinstance(corpus, corp.MaskedCorpus):

            sents = corpus.view_tokens(tok_name, unmask=True)

        else:

            sents = corpus.view_tokens(tok_name)

        for sent in sents:

            for i in xrange(sent.shape[0]):

                if corpus.terms[sent[i]] is not np.ma.masked:
                    
                    left = [env_matrix[term] for term in sent[:i]]

                    right = [env_matrix[term] for term in sent[i+1:]]
                    
                    sent_vecs = np.array(left + [psi] + right)
                    
                    conv_ngrams = reduce_ngrams(b_conv, sent_vecs, lmda, i)
                    
                    ord_vec = np.sum(conv_ngrams, axis=0)

                    self.matrix[sent[i], :] += ord_vec



class BeagleOrderMulti(model.Model):

    def train(self,
              corpus,
              env_matrix=None,
              psi=None,
              rand_perm=None,
              n_columns=2048,
              lmda = 7,
              tok_name='sentences',
              n_processes=20):

        global _lmda

        _lmda = lmda

        del lmda



        global _b_conv
        
        _b_conv = mk_b_conv(n_columns, rand_perm)

        del rand_perm



        if env_matrix is None:

            m = be.BeagleEnvironment()

            m.train(corpus, n_columns=n_columns)

            env_matrix = m.matrix[:]

            del m

        global _shape

        _shape = env_matrix.shape


        
        global _env_matrix

        print 'Copying env matrix to shared mp array'

        _env_matrix = mp.Array('f', env_matrix.size, lock=False)

        _env_matrix[:] = env_matrix.ravel()[:]

        del env_matrix



        global _psi

        _psi = mp.Array('f', _shape[1], lock=False)

        if psi is None:
            
            _psi[:] = rand_pt_unit_sphere(_shape[1])[:]

        else:

            _psi[:] = psi[:]

        del psi



        print 'Gathering tokens over which to map'

        if isinstance(corpus, corp.MaskedCorpus):

            sents = corpus.view_tokens(tok_name, unmask=True)

        else:

            sents = corpus.view_tokens(tok_name)

        k = len(sents) / (n_processes - 1)
        
        sent_lists = [sents[i * k:(i + 1) * k]
                      for i in xrange(n_processes - 1)]
        
        sent_lists.append(sents[(i + 1) * k:])
        
        tmp_dir = tempfile.mkdtemp()
        
        tmp_files = [os.path.join(tmp_dir, 'tmp_' + str(i))
                     for i in xrange(len(sent_lists))]

        sent_lists = [(sent_lists[i], tmp_files[i])
                      for i in xrange(len(sent_lists))]

        del sents



        global _terms

        _terms = corpus.terms

        del corpus


        
        try:

            # For debugging
            # tmp_files = map(mpfn, sent_lists)
            
            print 'Forking'
            
            p = mp.Pool()
            
            tmp_files = p.map(mpfn, sent_lists, 1)
            
            p.close()
            
            print 'Reducing'
            
            self.matrix = np.zeros(_shape)
            
            for filename in tmp_files:
                
                result = np.memmap(filename, mode='r',
                                   shape=_shape, dtype=np.float32)

                self.matrix[:, :] += result[:, :]

        finally:

            print 'Removing', tmp_dir

            shutil.rmtree(tmp_dir)



def mpfn((sents, filename)):

    n = _shape[1]

    result = np.memmap(filename, mode='w+', shape=_shape, dtype=np.float32)

    for sent in sents:

        for i,t in enumerate(sent):

            if _terms[sent[i]] is not np.ma.masked:

                left = [np.asarray(_env_matrix[term * n:(term + 1) * n])
                        for term in sent[:i]]
        
                right = [np.asarray(_env_matrix[term * n:(term + 1) * n])
                         for term in sent[i + 1:]]
        
                sent_vecs = np.array(left + [_psi] + right)
        
                conv_ngrams = reduce_ngrams(_b_conv, sent_vecs, _lmda, i)
            
                ord_vec = np.sum(conv_ngrams, axis=0)

                result[t] += ord_vec

    del result
    
    return filename



class BeagleOrder(BeagleOrderMulti):

    pass



#
# For testing
#



def test_BeagleOrderSingle():

    from vsm import corpus

    n = 5

    c = corpus.random_corpus(1e2, 10, 1, 10, tok_name='sentences')

    c = c.to_maskedcorpus()

    c.mask_terms(['0'])
    
    m = BeagleOrderSingle()

    m.train(c, n_columns=n)

    return c, m.matrix



def test_BeagleOrderMulti():

    from vsm import corpus

    n = 5

    print 'Generating corpus'
    
    c = corpus.random_corpus(1e2, 10, 1, 10, tok_name='sentences')

    c = c.to_maskedcorpus()

    c.mask_terms(['0'])

    m = BeagleOrderMulti()

    m.train(c, n_columns=n)

    return m.matrix



def test_compare():

    from vsm import corpus

    n = 4

    c = corpus.random_corpus(1e3, 20, 1, 10, tok_name='sentences')

    c = c.to_maskedcorpus()

    c.mask_terms(['0'])

    em = be.BeagleEnvironment()

    em.train(c, n_columns=n)

    env_matrix = em.matrix

    psi = rand_pt_unit_sphere(n)

    rand_perm = two_rand_perm(n)

    print 'Training single processor model'

    sm = BeagleOrderSingle()

    sm.train(c, psi=psi, env_matrix=env_matrix, rand_perm=rand_perm)

    print 'Training multiprocessor model'

    mm = BeagleOrderMulti()

    mm.train(c, psi=psi, env_matrix=env_matrix, rand_perm=rand_perm)

    assert np.allclose(sm.matrix, mm.matrix, atol=1e-07)




def test10():

    a = np.arange(5)

    def fn(x,y):

        if isinstance(x, tuple):

            return x + (y,)

        return (x, y)

    import pprint

    print 'array length', a.shape[0]
        
    for i in xrange(a.shape[0]):

        n = 3

        print 'ngram length', n

        print 'index', i

        pprint.pprint(reduce_ngrams(fn, a, n, i))

    for i in xrange(a.shape[0]):

        n = 4

        print 'ngram length', n

        print 'index', i

        pprint.pprint(reduce_ngrams(fn, a, n, i))

    for i in xrange(a.shape[0]):
                
        n = 5
            
        print 'ngram length', n

        print 'index', i
        
        pprint.pprint(reduce_ngrams(fn, a, n, i))



def test11():

    a = np.arange(5)

    def fn(x,y):

        return x + y

    import pprint

    print 'array length', a.shape[0]
        
    for i in xrange(a.shape[0]):

        n = 3

        print 'ngram length', n

        print 'index', i

        pprint.pprint(reduce_ngrams(fn, a, n, i))

    for i in xrange(a.shape[0]):

        n = 4

        print 'ngram length', n

        print 'index', i

        pprint.pprint(reduce_ngrams(fn, a, n, i))

    for i in xrange(a.shape[0]):
                
        n = 5
            
        print 'ngram length', n

        print 'index', i
        
        pprint.pprint(reduce_ngrams(fn, a, n, i))

