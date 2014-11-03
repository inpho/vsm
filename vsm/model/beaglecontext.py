import os
import shutil
import tempfile
import multiprocessing as mp
import cPickle as cpickle

import numpy as np

from base import BaseModel


__all__ = [ 'BeagleContextSeq', 'BeagleContextMulti' ]


def realign_env_mat(corpus, env_corpus, env_matrix):
    """
    """
    words = corpus.words
    indices = [env_corpus.words_int[w] for w in words]
    return env_matrix[indices]



class BeagleContextSeq(BaseModel):
    """

    """
    def __init__(self, corpus, env_corpus, env_matrix, 
                context_type='sentence'):
        """
        Initialize BeagleContextSeq.

        :param corpus: Source of observed data.
        :type corpus: class:`Corpus`

        :param env_corpus: BEAGLE environment corpus.
        :type env_corpus: class:`Corpus`

        :param env_matrix: Matrix from BEAGLE environment model.
        :type env_matrix: 2-D array

        :param context_type: Name of tokenization stored in `corpus` whose
            tokens will be treated as documents. Default is `sentence`.
        :type context_type: string, optional
        """
        self.context_type = context_type
        self.sents = corpus.view_contexts(context_type)
        self.env_matrix = realign_env_mat(corpus, env_corpus, env_matrix)


    def train(self):
        """
        Trains the model.
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
    """

    """

    def __init__(self, corpus, env_corpus, env_matrix, 
                 context_type='sentence'):
        """
        Initialize BeagleContextMulti.
        
        :param corpus: Souce of observed data.
        :type corpus: class:`Corpus`

        :param env_corpus: BEAGLE environment corpus. 
        :type env_corpus: class:`Corpus`

        :param env_matrix: Matrix from BEAGLE environment model.
        :type env_matrix: 2-D array

        :param context_type: Name of tokenization stored in `corpus` whose
            tokens will be treated as documents. Default is `sentence`.
        :type context_type: string, optional
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
        Takes an optional argument `n_procs`, number of processors,
        and trains the model on the number of processors. `n_procs`
        is 2 by default.
        
        :param n_procs: Number of processors. Default is 2.
        :type n_procs: int, optional

        :returs: `None`
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

