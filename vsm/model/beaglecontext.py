import os
import shutil
import tempfile
import multiprocessing as mp

import numpy as np

from vsm import corpus as corp
from vsm import model
from vsm.model import beagleenvironment as be



class BeagleContextSingle(model.Model):

    def train(self,
              corpus,
              env_matrix=None,
              n_columns=2048,
              tok_name='sentences'):

        if env_matrix is None:

            m = be.BeagleEnvironment()

            m.train(corpus, n_columns=n_columns)

            env_matrix = m.matrix[:, :]

        self.matrix = np.zeros_like(env_matrix)



        if isinstance(corpus, corp.MaskedCorpus):

            sents = corpus.view_tokens(tok_name, compress=True)

        else:

            sents = corpus.view_tokens(tok_name)



        for sent in sents:

            if sent.shape[0] > 1:

                left_sums = np.cumsum(env_matrix[sent[:-1]], axis=0)

                right_sums = np.cumsum(env_matrix[sent[:0:-1]], axis=0)

                for i,term in enumerate(sent):

                    if i == 0:

                        ctx_vector = right_sums[-1]

                    elif i == sent.shape[0] - 1:
                        
                        ctx_vector = left_sums[-1]
                    
                    else:

                        ctx_vector = left_sums[i - 1] + right_sums[-i - 1]

                    self.matrix[term, :] += ctx_vector



class BeagleContextMulti(model.Model):

    def train(self,
              corpus,
              env_matrix=None,
              n_columns=2048,
              tok_name='sentences',
              n_processes=20):

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



        print 'Gathering tokens over which to map'

        if isinstance(corpus, corp.MaskedCorpus):

            sents = corpus.view_tokens(tok_name, compress=True)

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
        
        if sent.shape[0] > 1:

            lslice = slice(sent[0] * n, (sent[0] + 1) * n)

            left_sums = [np.array(_env_matrix[lslice])]

            rslice = slice(sent[-1] * n, (sent[-1] + 1) * n)

            right_sums = [np.array(_env_matrix[rslice])]

            for i in xrange(1, sent.shape[0] - 1):

                lslice = slice(sent[i] * n, (sent[i] + 1) * n)

                next_left = np.array(_env_matrix[lslice])

                left_sums.append(left_sums[-1] + next_left)

                rslice = slice(sent[-i - 1] * n, (sent[-i - 1] + 1) * n)

                next_right = np.array(_env_matrix[rslice])

                right_sums.append(right_sums[-1] + next_right)

            for i,t in enumerate(sent):
                
                if i == 0:

                    ctx_vector = right_sums[-1]

                elif i == sent.shape[0] - 1:
                        
                    ctx_vector = left_sums[-1]
                    
                else:

                    ctx_vector = left_sums[i - 1] + right_sums[-i - 1]
                    
                result[t, :] += ctx_vector

    del result
    
    return filename




class BeagleContext(BeagleContextMulti):

    pass



#
# For testing
#



def test_BeagleContextSingle():

    from vsm import corpus

    n = 5

    c = corpus.random_corpus(1e2, 10, 1, 10, tok_name='sentences')

    c = c.to_maskedcorpus()

    c.mask_terms(['0'])
        
    m = BeagleContextSingle()

    m.train(c, n_columns=n)

    return m.matrix



def test_BeagleContextMulti():

    from vsm import corpus

    n = 5

    print 'Generating corpus'
    
    c = corpus.random_corpus(1e2, 10, 1, 10, tok_name='sentences')

    c = c.to_maskedcorpus()

    c.mask_terms(['0'])
    
    m = BeagleContextMulti()

    m.train(c, n_columns=n)

    return m.matrix



def test_compare():

    from vsm import corpus

    n = 4

    c = corpus.random_corpus(1e3, 20, 1, 10, tok_name='sentences')

    em = be.BeagleEnvironment()

    em.train(c, n_columns=n)

    env_matrix = em.matrix

    print 'Training single processor model'

    sm = BeagleContextSingle()

    sm.train(c, env_matrix=env_matrix)

    print 'Training multiprocessor model'

    mm = BeagleContextMulti()

    mm.train(c, env_matrix=env_matrix)

    assert np.allclose(sm.matrix, mm.matrix, atol=1e-07), (sm.matrix[:2], mm.matrix[:2])
