import numpy as np

from vsm.corpus import Corpus


__all__=['corpus_fromlist']



def corpus_fromlist(ls, context_type='context'):
    """
    Takes a list of lists or arrays containing strings or integers and
    returns a Corpus object. The label associated to a given context
    is `context_type` prepended to the context index.
    """
    corpus = [w for ctx in ls for w in ctx]

    indices = np.cumsum([len(sbls) for sbls in ls])
    metadata = ['{0}_{1}'.format(context_type, i)
                for i in xrange(len(indices))]
    md_type = np.array(metadata).dtype
    dtype = [('idx', np.int), (context_type + '_label', md_type)]
    context_data = [np.array(zip(indices, metadata), dtype=dtype)]

    return Corpus(corpus, context_data=context_data,
                  context_types=[context_type])
