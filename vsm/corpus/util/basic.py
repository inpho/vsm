import numpy as np

from vsm.corpus import Corpus


__all__=['empty_corpus', 'random_corpus', 'corpus_fromlist']


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


def empty_corpus(context_type='context'):
    """
    """
    return Corpus([],
                  context_data=[np.array([], dtype=[('idx', np.int)])],
                  context_types=[context_type])


def random_corpus(corpus_len,
                  n_words,
                  min_token_len,
                  max_token_len,
                  context_type='context',
                  metadata=False):
    """
    Generate a random integer corpus.
    """
    corpus = np.random.randint(n_words, size=corpus_len)

    indices = []
    i = np.random.randint(min_token_len, max_token_len)
    while i < corpus_len:
        indices.append(i)
        i += np.random.randint(min_token_len, max_token_len)
    indices.append(corpus_len)

    if metadata:
        metadata_ = ['{0}_{1}'.format(context_type, i)
                     for i in xrange(len(indices))]
        dtype=[('idx', np.array(indices).dtype), 
               (context_type + '_label', np.array(metadata_).dtype)]
        rand_tok = np.array(zip(indices, metadata_), dtype=dtype)
    else:
        rand_tok = np.array([(i,) for i in indices], 
                            dtype=[('idx', np.array(indices).dtype)])

    return Corpus(corpus, context_types=[context_type], context_data=[rand_tok])
