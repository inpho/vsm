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
    the user the sequental algorithm is being used.
    """
    def __new__(cls,
                corpus=None, context_type=None,
                K=20, V=0, alpha=[], beta=[],
                multiprocessing=False):

        kwargs = dict(corpus=corpus, context_type=context_type,
                      K=K, V=V, alpha=alpha, beta=beta)
        
        if multiprocessing and platform.system() != 'Windows':
            return LdaCgsMulti(**kwargs)
        else:
            if platform.system() == 'Windows':
                warnings.warn("""Multiprocessing is not implemented on Windows.
                Defaulting to sequential algorithm.""", RuntimeWarning)
            return LdaCgsSeq(**kwargs)

    @staticmethod
    def load(filename, multiprocessing=False):
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
