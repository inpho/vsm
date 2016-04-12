import numpy as np
import time
from vsm.split import split_corpus
from vsm.corpus import align_corpora as align
from ldafunctions import *
from _cgs_update import cgs_update
from progressbar import ProgressBar, Percentage, Bar


__all__ = [ 'LdaCgsSeq', 'LdaCgsQuerySampler' ]



class LdaCgsSeq(object):
    """
    An implementation of LDA using collapsed Gibbs sampling.
    """
    def __init__(self, corpus=None, context_type=None,
                 K=20, V=0, alpha=[], beta=[], seed=None):
        """
        Initialize LdaCgsSeq.

        :param corpus: Source of observed data.
        :type corpus: `Corpus`

        :param context_type: Name of tokenization stored in `corpus` whose tokens
            will be treated as documents.
        :type context_type: string, optional

        :param K: Number of topics. Default is `20`.
        :type K: int, optional

        :param alpha: Document priors. Default is a flat prior of 0.01
            for all topics.
        :type alpha: list, optional

        :param beta: Topic priors. Default is 0.01 for all words.
        :type beta: list, optional

        :param seed: Seed for numpy's RandomState. Default is `None`.
        :type seed: int, optional
        """

        self.context_type = context_type
        self.K = K

        if corpus:
            self.V = corpus.words.size
            self.indices = corpus.view_contexts(self.context_type, as_indices=True)
            self.corpus = corpus.corpus
        else:
            self.V = V
            self.indices = []
            self.corpus = []

        self.indices = np.array(self.indices, dtype='i')
        self.Z = np.zeros_like(self.corpus, dtype='i')

        priors = init_priors(self.V, self.K, beta, alpha)
        self.beta, self.alpha = priors

        self.word_top = (np.zeros((self.V, self.K), dtype=np.float)
                         + self.beta)
        if self.V==0:
            self.inv_top_sums = np.inf
        else:
            self.inv_top_sums = 1. / self.word_top.sum(0)
        self.top_doc = (np.zeros((self.K, len(self.indices)),
                                 dtype=np.float) + self.alpha)

        self.iteration = 0
        self.log_probs = []

        if seed is None:
            maxint = np.iinfo(np.int32).max
            self.seed = np.random.randint(0, maxint)
        else:
            self.seed = seed
        self._mtrand_state = np.random.RandomState(self.seed).get_state()


    @property
    def Z_split(self):
        return split_corpus(self.Z, self.indices)


    @property
    def docs(self):
        return split_corpus(self.corpus, self.indices)


    def _compute_word_top(self):
        self.word_top = compute_word_top(self.docs, self.Z_split, self.K,
                                         self.V, self.beta)


    def _compute_top_doc(self):
        self.top_doc = compute_top_doc(self.Z_split, self.K, self.alpha)


    def _compute_log_prob(self, increment=False):
        log_prob = compute_log_prob(self.docs, self.Z_split,
                                    self.word_top, self.top_doc)
        if increment:
            self.log_probs.append((self.iteration, log_prob))
            self.iteration += 1
        else:
            return log_prob


    def train(self, n_iterations=100, verbose=1, **kwargs):
        """
        Takes an optional argument, `n_iterations` and updates the model
        `n_iterations` times.

        :param n_iterations: Number of iterations. Default is 100.
        :type n_iterations: int, optional

        :param verbose: If 1, current number of iterations
            are printed out to notify the user. Default is 1.
        :type verbose: int, optional
        
        :param kwargs: For compatability with calls to LdaCgsMulti.
        :type kwargs: optional
        """
        random_state = np.random.RandomState(self.seed)
        random_state.set_state(self._mtrand_state)


        if verbose > 0:
            print ('Begin LDA training for {0} iterations'\
                   .format(n_iterations))
            start = time.time()
            t = start

        # Training loop
        stop = self.iteration + n_iterations
        if verbose == 1:
            pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=n_iterations).start()

        #print("Stop ", stop)
        for itr in xrange(self.iteration , stop):

            results = cgs_update(self.iteration, self.corpus, self.word_top,
                                 self.inv_top_sums, self.top_doc, self.Z,
                                 self.indices, self._mtrand_state[0],
                                 self._mtrand_state[1], self._mtrand_state[2],
                                 self._mtrand_state[3], self._mtrand_state[4])

            lp = results[4]
            self.log_probs.append((self.iteration, lp))

            if verbose == 2:
                itr_time = np.around(time.time()-t, decimals=1)
                t = time.time()
                if verbose > 1 or itr==stop-1:
                    print ('\nIteration {0} complete: log_prob={1}, time={2}'
                           .format(self.iteration, lp, itr_time))

            if verbose == 1:
                #print("Self iteration", self.iteration)
                pbar.update(self.iteration - (stop - n_iterations))
                time.sleep(0.01)

            self.iteration += 1

            self._mtrand_state = results[5:]
        if verbose == 1:
            pbar.finish();
        if verbose > 1:
            print '-'*60, ('\n\nWalltime per iteration: {0} seconds'
                           .format(np.around((t-start)/n_iterations, decimals=2)))


    @staticmethod
    def load(filename):
        """
        A static method for loading a saved LdaCgsMulti model.

        :param filename: Name of a saved model to be loaded.
        :type filename: string

        :returns: m : LdaCgsMulti object

        :See Also: :class:`numpy.load`
        """
        return load_lda(filename, LdaCgsSeq)


    def save(self, filename):
        """
        Saves the model in an `.npz` file.

        :param filename: Name of a saved model to be loaded.
        :type filename: string

        :See Also: :class:`numpy.savez`
        """
        save_lda(self, filename)



class LdaCgsQuerySampler(LdaCgsSeq):
    """
    """
    def __init__(self, lda_obj=None, new_corpus=None,
                 align_corpora=False, old_corpus=None,
                 context_type=None):

        if align_corpora:
            new_corp = align(old_corpus, new_corpus)

        if lda_obj:
            if context_type is None:
                context_type = lda_obj.context_type

            kwargs = dict(corpus=new_corpus,
                          context_type=context_type,
                          K=lda_obj.K, V=lda_obj.V,
                          alpha=lda_obj.alpha, beta=lda_obj.beta)
        else:
            kwargs = dict(corpus=new_corpus)


        super(LdaCgsQuerySampler, self).__init__(**kwargs)

        if lda_obj:
            self.word_top[:] = lda_obj.word_top
            self.inv_top_sums[:] = lda_obj.inv_top_sums




#################################################################
#                            Demos
#################################################################


def demo_LdaCgsSeq(doc_len=500, V=100000, n_docs=100,
                   K=20, n_iterations=5,
                   corpus_seed=None, model_seed=None):

    from vsm.extensions.corpusbuilders import random_corpus

    print 'Words per document:', doc_len
    print 'Words in vocabulary:', V
    print 'Documents in corpus:', n_docs
    print 'Number of topics:', K
    print 'Iterations:', n_iterations

    c = random_corpus(n_docs*doc_len, V, doc_len, doc_len+1, seed=corpus_seed)
    m = LdaCgsSeq(c, 'document', K=K, seed=model_seed)
    m.train(n_iterations=n_iterations, verbose=2)

    return m
