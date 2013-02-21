import os
import shutil
import tempfile
import multiprocessing as mp
import cPickle as cpickle

import numpy as np

from vsm.model import BaseModel



def realign_env_mat(corpus, env_corpus, env_matrix):
    """
    """
    words = corpus.words
    indices = [env_corpus.words_int[w] for w in words]
    return env_matrix[indices]



class BeagleContextSeq(BaseModel):

    def __init__(self, corpus, env_corpus, env_matrix, 
                context_type='sentence'):
        """
        """
        self.context_type = context_type
        self.sents = corpus.view_contexts(context_type)
        self.env_matrix = realign_env_mat(corpus, env_corpus, env_matrix)


    def train(self):
        """
        """
        self.matrix = np.zeros_like(self.env_matrix)

        for sent in self.sents:

            if sent.shape[0] > 1:

                left_sums = np.cumsum(self.env_matrix[sent[:-1]], axis=0)
                right_sums = np.cumsum(self.env_matrix[sent[:0:-1]], axis=0)

                for i,word in enumerate(sent):

                    if i == 0:
                        ctx_vector = right_sums[-1]

                    elif i == sent.shape[0] - 1:
                        ctx_vector = left_sums[-1]
                    
                    else:
                        ctx_vector = left_sums[i - 1] + right_sums[-i - 1]

                    self.matrix[word, :] += ctx_vector



class BeagleContextMulti(BaseModel):

    def __init__(self, corpus, env_corpus, env_matrix, 
                 context_type='sentence'):
        """
        """
        self.context_type = context_type
        self.sents = corpus.view_contexts(context_type)
        self.dtype = env_matrix.dtype
        env_matrix = realign_env_mat(corpus, env_corpus, env_matrix)

        global _shape 
        _shape = mp.Array('i', 2, lock=False)
        _shape[:] = env_matrix.shape
        
        print 'Copying env matrix to shared mp array'
        global _env_matrix
        _env_matrix = mp.Array('d', env_matrix.size, lock=False)
        _env_matrix[:] = env_matrix.ravel()[:]


    def train(self, n_procs=2):
        """
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
        if sent.shape[0] > 1:

            env = np.empty((sent.size, _shape[1]), dtype=np.float64)
            for i,w in enumerate(sent):
                env[i, :] = _env_matrix[w*_shape[1]: (w+1)*_shape[1]]
        
            left_sums = np.cumsum(env[:-1], axis=0)
            right_sums = np.cumsum(env[:0:-1], axis=0)

            for i,t in enumerate(sent):
                
                if i == 0:
                    ctx_vector = right_sums[-1]

                elif i == sent.shape[0] - 1:
                    ctx_vector = left_sums[-1]
                    
                else:
                    ctx_vector = left_sums[i - 1] + right_sums[-i - 1]
                   
                if t in result:
                    result[t] += ctx_vector
                else:
                    result[t] = ctx_vector

    with open(filename, 'wb') as f:
        cpickle.dump(result, f)

    return filename




#
# For testing
#



def test_BeagleContextSeq():

    from vsm.util.corpustools import random_corpus
    from vsm.model.beagleenvironment import BeagleEnvironment

    ec = random_corpus(100000, 5000, 0, 20, context_type='sentence')
    cc = ec.apply_stoplist(stoplist=[str(i) for i in xrange(0,5000,100)])

    e = BeagleEnvironment(ec, n_cols=200)
    e.train()

    m = BeagleContextSeq(cc, ec, e.matrix)
    m.train()

    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m.save(tmp.name)
        tmp.close()
        m1 = BeagleContextSeq.load(tmp.name)
        assert (m.matrix == m1.matrix).all()
    
    finally:
        os.remove(tmp.name)

    return m.matrix


def test_BeagleContextMulti():

    from vsm.util.corpustools import random_corpus
    from vsm.model.beagleenvironment import BeagleEnvironment

    ec = random_corpus(100000, 5000, 0, 20, context_type='sentence')
    cc = ec.apply_stoplist(stoplist=[str(i) for i in xrange(0,5000,100)])

    e = BeagleEnvironment(ec, n_cols=200)
    e.train()

    m = BeagleContextMulti(cc, ec, e.matrix)
    m.train(n_procs=3)

    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m.save(tmp.name)
        tmp.close()
        m1 = BeagleContextMulti.load(tmp.name)
        assert (m.matrix == m1.matrix).all()
    
    finally:
        os.remove(tmp.name)

    return m.matrix


def test_compare():

    from vsm.util.corpustools import random_corpus
    from vsm.model.beagleenvironment import BeagleEnvironment

    ec = random_corpus(100000, 5000, 0, 20, context_type='sentence')
    cc = ec.apply_stoplist(stoplist=[str(i) for i in xrange(0,5000,100)])

    e = BeagleEnvironment(ec, n_cols=5)
    e.train()

    print 'Training single processor model'
    ms = BeagleContextSeq(cc, ec, e.matrix)
    ms.train()

    print 'Training multiprocessor model'
    mm = BeagleContextMulti(cc, ec, e.matrix)
    mm.train()

    assert np.allclose(ms.matrix, mm.matrix)
