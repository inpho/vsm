import numpy as np

from vsm.corpus import Corpus



__all__ = ['random_corpus']



def random_corpus(corpus_len,
                  n_words,
                  min_token_len,
                  max_token_len,
                  context_type='random',
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
