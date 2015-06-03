"""
Provides a convenient alias for the LdaCgs* classes
"""
import platform # For Windows workaround
import warnings

from ldacgsseq import *
from ldacgsmulti import *
from ldafunctions import load_lda


__all__ = [ 'LDA' ]


class LDA(object):
    """
    Depending on the boolean parameter `multiprocessing`, returns and
    initializes an instance of either LdaCgsSeq or LdaCgsMulti.

    Note that on Windows platforms, `multiprocessing` is not implemented.
    In contrast to LdaCgsMulti, LDA always returns a valid object. Instead
    of raising a NotImplementedError, LDA issues a RuntimeWarning, notifying 
    the user the sequental algorithm is being used. When `seed_or_seeds` is a
    list in this instance, only the first seed is used. 
    """
    def __new__(cls,
                corpus=None, context_type=None,
                K=20, V=0, alpha=[], beta=[],
                multiprocessing=False, seed_or_seeds=None, n_proc=None):

        kwargs = dict(corpus=corpus, context_type=context_type,
                      K=K, V=V, alpha=alpha, beta=beta)
        
        if multiprocessing and platform.system() != 'Windows':
            if n_proc is not None:
                kwargs['n_proc'] = n_proc
            if seed_or_seeds is not None and not isinstance(seed_or_seeds, int):
                kwargs['seeds'] = seed_or_seeds

            return LdaCgsMulti(**kwargs)

        else:
            if multiprocessing and platform.system() == 'Windows':
                warnings.warn("""Multiprocessing is not implemented on Windows.
                Defaulting to sequential algorithm.""", RuntimeWarning)
                
                # extract single seed
                if seed_or_seeds is not None and not isinstance(seed_or_seeds, int):
                    seed_or_seeds = seed_or_seeds[0]
                    warnings.warn("Windows is using only the first seed: " +
                                  str(seed_or_seeds), RuntimeWarning)

            # parse seed_or_seeds argument
            if isinstance(seed_or_seeds, int):
                kwargs['seed'] = seed_or_seeds
            elif seed_or_seeds is not None:
                raise ValueError("LDA(seed_or_seeds, ...) must take an" +
                                 "integer in single-threaded mode.")

            return LdaCgsSeq(**kwargs)

    @staticmethod
    def load(filename, multiprocessing=False, n_proc=None):
        """
        A static method for loading a saved LdaCgsMulti model.

        :param filename: Name of a saved model to be loaded.
        :type filename: string

        :returns: m : LdaCgsMulti object

        :See Also: :class:`numpy.load`
        """
        if multiprocessing and platform.system() != 'Windows':
            return load_lda(filename, LdaCgsMulti)
        else:
            if platform.system() == 'Windows':
                warnings.warn("""Multiprocessing is not implemented on Windows.
                Defaulting to sequential algorithm.""", RuntimeWarning)
            return load_lda(filename, LdaCgsSeq)
