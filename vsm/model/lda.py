"""
Provides a convenient alias for the LdaCgs* classes
"""

from ldacgsseq import *
from ldacgsmulti import *


__all__ = [ 'LDA' ]


class LDA(object):
    """
    Depending on the boolean parameter `multiprocessing`, returns and
    initializes an instance of either LdaCgsSeq or LdaCgsMulti.
    """
    def __new__(cls,
                corpus=None, context_type=None,
                K=20, V=0, alpha=[], beta=[],
                multiprocessing=False):

        kwargs = dict(corpus=corpus, context_type=context_type,
                      K=K, V=V, alpha=alpha, beta=beta)
        
        if multiprocessing:
            return LdaCgsMulti(**kwargs)
        else:
            return LdaCgsSeq(**kwargs)
