import os
import shutil
import tempfile
import multiprocessing as mp
import cPickle as cpickle

import numpy as np
from numpy import dual

from vsm.spatial import rand_pt_unit_sphere
from base import BaseModel


__all__ = [ 'BeagleOrderSeq', 'BeagleOrderMulti' ]



def two_rand_perm(n, seed=None):
    """
    """
    np.random.seed(seed)
    
    perm1 = np.random.permutation(n)

    while True:
        perm2 = np.random.permutation(n)

        if not (perm2 == perm1).all():
            break
        
    return perm1, perm2


def mk_b_conv(n, rand_perm=None):
    """
    """
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


class BeagleOrderSeq(BaseModel):
    """
    `BeagleOrderSeq` stores word order information in the context.
    """

    def __init__(self, corpus, env_matrix, context_type='sentence',
                 psi=None, rand_perm=None, lmda =7):
        """ 
        Initialize BeagleOrderSeq.
        
        :param corpus: Soure of observed data.
        :type corpus: Corpus

        :param env_matrix: BEAGLE environment matrix.
        :type env_matrix: 2-D array

        :param context_type: Name of tokenization stored in `corpus` whose
            tokens will be treated as documents. Default is `sentence`.
        :type context_type: string, optional

        :param psi:  
        :type psi: int, optional

        :param rand_perm:  
        :type rand_perm: boolean, optional

        :param lmda:  
        :type lmda: int, optional
        """
        self.context_type = context_type
        self.sents = corpus.view_contexts(context_type)
        self.env_matrix = env_matrix

        self.b_conv = mk_b_conv(env_matrix.shape[1], rand_perm)
        
        if psi is None:
            self.psi = rand_pt_unit_sphere(env_matrix.shape[1])
        else:
            self.psi = psi

        self.lmda = lmda


    def train(self):
        """
        Trains the model.
        """
        self.matrix = np.zeros_like(self.env_matrix)

        for sent in self.sents:
            env = self.env_matrix[sent]
            
            for i in xrange(sent.size):
                sent_vecs = env.copy() 
                sent_vecs[i, :] = self.psi[:]

                conv_ngrams = reduce_ngrams(self.b_conv, sent_vecs, self.lmda, i)
                    
                ord_vec = np.sum(conv_ngrams, axis=0)
                self.matrix[sent[i], :] += ord_vec



class BeagleOrderMulti(BaseModel):
    """
    `BeagleOrderSeq` stores word order information in the context.
    """
    def __init__(self, corpus, env_matrix, context_type='sentence',
                 psi=None, rand_perm=None, lmda =7):
        """    
        Initialize BeagleOrderMulti.

        :param corpus: Source of observed data.
        :type corpus: Corpus

        :param env_matrix: BEAGLE environement matrix.
        :type env_matrix: 2-D array

        :param context_type: Name of tokenization stored in `corpus` whose
            tokens will be treated as documents. Default is `sentence`.
        :type context_type: string, optional

        :param psi:  
        :type psi: int, optional

        :param rand_perm:  
        :type rand_perm: boolean, optional

        :param lmda:  
        :type lmda: int, optional
        """
        self.context_type = context_type
        self.sents = corpus.view_contexts(context_type)
        self.dtype = env_matrix.dtype

        global _shape 
        _shape = mp.Array('i', 2, lock=False)
        _shape[:] = env_matrix.shape

        print 'Copying env matrix to shared mp array'
        global _env_matrix
        _env_matrix = mp.Array('d', env_matrix.size, lock=False)
        _env_matrix[:] = env_matrix.ravel()[:]

        #TODO: Convert this to a shared data structure
        global _b_conv 
        _b_conv = mk_b_conv(env_matrix.shape[1], rand_perm)

        global _psi        
        _psi = mp.Array('d', _shape[1], lock=False)
        if psi is None:
            _psi[:] = rand_pt_unit_sphere(env_matrix.shape[1])[:]
        else:
            _psi[:] = psi[:]

        global _lmda
        _lmda = mp.Value('i', lock=False)
        _lmda =lmda


    def train(self, n_procs=2):
        """
        Trains the model using `n_procs` processors.

        :param n_procs: Number of processors. Default is 2.
        :type n_procs: int, optional
        """
        sent_lists = np.array_split(self.sents, n_procs-1)
        if len(sent_lists) != n_procs:
            sent_lists = np.array_split(self.sents, n_procs)

        tmp_dir = tempfile.mkdtemp()
        tmp_files = [os.path.join(tmp_dir, 'tmp_' + str(i))
                     for i in xrange(len(sent_lists))]

        sent_lists = zip(sent_lists, tmp_files)
        del self.sents


        try:
            print 'Forking'
            # For debugging
            # tmp_files = map(mpfn, sent_lists)
            
            p = mp.Pool(n_procs)
            tmp_files = p.map(mpfn, sent_lists, 1)
            p.close()

            print 'Reducing'
            self.matrix = np.zeros(tuple(_shape), dtype=self.dtype)

            for filename in tmp_files:

                with open(filename, 'rb') as f:
                    result = cpickle.load(f)

                for k,v in result.iteritems():
                    self.matrix[k, :] += v

        finally:
            print 'Removing', tmp_dir
            shutil.rmtree(tmp_dir)
        

def mpfn((sents, filename)):
    """
    """
    result = dict()

    for sent in sents:

        env = np.empty((sent.size, _shape[1]), dtype=np.float64)
        for i,w in enumerate(sent):
            env[i, :] = _env_matrix[w*_shape[1]: (w+1)*_shape[1]]

        for i,w in enumerate(sent):
            sent_vecs = env.copy()
            sent_vecs[i, :] = _psi[:]

            conv_ngrams = reduce_ngrams(_b_conv, sent_vecs, _lmda, i)

            ord_vec = np.sum(conv_ngrams, axis=0)

            if w in result:
                result[w] += ord_vec
            else:
                result[w] = ord_vec
            
    with open(filename, 'wb') as f:
        cpickle.dump(result, f)

    return filename
