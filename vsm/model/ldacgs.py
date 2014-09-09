import numpy as np
import time
from ldafunctions import load_lda, save_lda, init_priors

# import pyximport; pyximport.install()
from _ldacgs import cgs


__all__ = [ 'LdaCgs' ]



class LdaCgs(object):
    """
    """
    def __init__(self, corpus=None, context_type=None,
                 K=20, V=0, alpha=[], beta=[]):
        """
        Initialize LdaCgs.

        :param corpus: Source of observed data.
        :type corpus: `Corpus`
    
        :param context_type: Name of tokenization stored in `corpus` whose tokens
            will be treated as documents.
        :type context_type: string, optional

        :param K: Number of topics. Default is `20`.
        :type K: int, optional
    
        :param beta: Topic priors. Default is 0.01 for all words.
        :type beta: list, optional
    
        :param alpha: Document priors. Default is a flat prior of 0.01 
            for all topics.
        :type alpha: list, optional
        """

        self.context_type = context_type
        self.K = K

        if corpus:
            self.V = corpus.words.size
            self.indices = corpus.view_contexts(self.context_type, 
                                                as_indices=True)
            self.indices = np.array(self.indices, dtype=('i'))
            self.corpus = np.array(corpus.corpus, dtype=('i'))
        else:
            self.V = V
            self.indices = np.array([], dtype=('i'))
            self.corpus = np.array([], dtype=('i'))

        priors = init_priors(self.V, self.K, beta, alpha)
        self.beta, self.alpha = priors

        self.Z = None
        self.word_top = None
        self.top_doc = None

        self.log_probs = None
        self.iteration = 0


    def train(self, n_iterations=100, n_threads=1, verbose=1):

        seed = np.uint64(0)

        results = cgs(self.K, 
                      self.V,
                      self.indices,
                      self.corpus,
                      self.alpha.reshape(-1,),
                      self.beta.reshape(-1,),
                      n_iterations,
                      n_threads,
                      seed)

        self.Z = results['Z']
        self.word_top = results['word_top']
        self.top_doc = results['top_doc']
        #TODO: Manage log_probs so that training continuations can be done.
        self.log_probs = results['log_probs']


    @staticmethod
    def load(filename):
        return load_lda(filename, LdaCgsSeq)


    def save(self, filename):
        save_lda(self, filename)



#################################################################
#                            Demos
#################################################################


def demo_LdaCgs(doc_len=500, V=100000, n_docs=100,
                K=20, n_iterations=5, n_threads=1):

    from vsm.extensions.corpusbuilders import random_corpus
    
    print 'Words per document:', doc_len
    print 'Words in vocabulary:', V
    print 'Documents in corpus:', n_docs
    print 'Number of topics:', K
    print 'Iterations:', n_iterations

    c = random_corpus(n_docs*doc_len, V, doc_len, doc_len+1)

    print 'Random corpus generated. Initializing model.'
    m = LdaCgs(c, 'document', K=K)
    
    print 'Begin estimation.'
    m.train(n_iterations=n_iterations, n_threads=n_threads)

    return m
