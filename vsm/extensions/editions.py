import numpy as np
from vsm.corpus import Corpus


__all__ = ['eqva', 'new_material']


def eqva(a1, a2):
    """
    modified np.array_equal. considers a1 and a2
    equal when there is 1 difference.
    """
    a1.sort()
    a2.sort()
    count = 0
    a1_, a2_ = a1, a2
    if len(a1) > len(a2):
        a1_ = a2
        a2_ = a1

    for s in a1:
        if not s in a2:
            count += 1

    return count


def find_idx(ind, c1, c2):
    """
    finds exact match (1 diff) in c2 and returns the index.
    """
    ctx2 = c2.view_contexts('sentence', as_strings=True)
    ctx = c1.view_contexts('sentence', as_strings=True)[ind]
    
    for i in xrange(len(ctx2)):
        if eqva(ctx, ctx2[i]) < 2:
            return str(i)
    return ''


def new_material(c1, c2, idx=0):
    """
    Return new material in a list. 
    'idx' is an optional parameter for cutting off references.
    """
    ctx1 = c1.view_contexts('sentence', as_strings=True)
    
    if idx == 0:
        ctx2 = c2.view_contexts('sentence', as_strings=True)
    else:
        ctx2 = c2.view_contexts('sentence', as_strings=True)[:idx]
    len2 = len(ctx2)

    new = []
    for i in xrange(len(ctx1)):
        if i < len2:
            if len(ctx1[i]) == 0: # empty tokens.
                pass
            else:
                ind = find_idx(i, c1, c2)
                if len(ind) == 0:
                    new.append(i)
    return new
