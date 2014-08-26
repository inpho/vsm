import numpy as np


__all__ = [ 'init_priors', 'categorical', 'cgs_update',
            'load_lda', 'save_lda' ]



def init_priors(V=0, K=0, beta=[], alpha=[]):

    # Topic and context priors; set defaults if need be
    if len(beta) > 0:
        beta = (np.array(beta, dtype=np.float).reshape(len(beta), 1))
    else:
        # Default is a flat prior of .01
        beta = np.ones((V, 1), dtype=np.float) * .01

    if len(alpha) > 0:
        alpha = (np.array(alpha, dtype=np.float).reshape(len(alpha), 1))
    else:
        # Default is a flat prior of .01
        alpha = np.ones((K, 1), dtype=np.float) * .01

    return beta, alpha


def categorical(pvals, random_state=None):
    """
    Draws a sample from the categorical distribution parameterized by
    `pvals`.
    """
    if not random_state:
        random_state = np.random.RandomState()
    cum_dist = np.cumsum(pvals)
    r = random_state.uniform() * cum_dist[-1]
    return np.searchsorted(cum_dist, r)


def cgs_update(itr, docs, word_top, inv_top_sums, 
               top_doc, Z, random_state=None):

    log_p = 0
    log_wk = np.log(word_top * inv_top_sums[np.newaxis, :])
    log_kd = np.log(top_doc / top_doc.sum(0)[np.newaxis, :])

    for i in xrange(len(docs)):
        for j in xrange(len(docs[i])):
            
            w,k = docs[i][j], Z[i][j]

            log_p += log_wk[w, k] + log_kd[k, i]

            if itr > 0:
                word_top[w, k] -= 1
                inv_top_sums[k] *= 1. / (1 - inv_top_sums[k])
                top_doc[k, i] -= 1

            dist = inv_top_sums * word_top[w,:] * top_doc[:,i]

            k = categorical(dist, random_state=random_state)

            word_top[w, k] += 1
            inv_top_sums[k] *= 1. / (1 + inv_top_sums[k]) 
            top_doc[k, i] += 1

            Z[i][j] = k

    return word_top, inv_top_sums, top_doc, Z, log_p


def compute_top_doc(Z, K, alpha=[]):
    """
    Takes a topic assignment Z, the number of topics K and optionally
    a document prior and returns the topic-document matrix which is
    used in the LDA model objects.
    """    
    if len(alpha)==0:
        top_doc = np.zeros((K, len(Z)), dtype=np.float)
    else:
        top_doc = np.zeros((K, len(Z)), dtype=np.float) + alpha

    for i in xrange(len(Z)):
        for j in xrange(len(Z[i])):
            z = Z[i][j]
            top_doc[z][i] += 1

    return top_doc


def compute_word_top(W, Z, K, V, beta=[]):
    """
    Takes a list of documents W, a topic assignment Z, the number of
    topics K, the number of words in the vocabulary V and optionally a
    topic prior and returns the word-topic matrix which is used in the
    LDA model objects.
    """    
    if len(beta)==0:
        word_top = np.zeros((V, K), dtype=np.float)
    else:
        word_top = np.zeros((V, K), dtype=np.float) + beta

    for i in xrange(len(Z)):
        for j in xrange(len(Z[i])):
            w = W[i][j]
            k = Z[i][j]
            word_top[w][k] += 1

    return word_top

    
def load_lda(filename, ldaclass):
    """
    A static method for loading a saved `ldaclass` model.
    
    :param filename: Name of a saved model to be loaded.
    :type filename: string
    
    :returns: m : `ldaclass` object
    
    :See Also: :class:`numpy.load`
    """
    print 'Loading LDA data from', filename
    arrays_in = np.load(filename)
    try:
        context_type = arrays_in['context_type'][()]
        K = arrays_in['K'][()]
        alpha = arrays_in['alpha']
        beta = arrays_in['beta']
    
        m = ldaclass(context_type=context_type, K=K,
                     alpha=alpha, beta=beta)

        m.V = arrays_in['V'][()]
        m.corpus = arrays_in['corpus']

        m.Z_indices = ['Z_indices']
        m.Z_flat = arrays_in['Z_flat']
        
        m.iteration = arrays_in['iteration'][()]
        m.log_probs = arrays_in['log_probs'].tolist()

        m.top_doc = arrays_in['top_doc']
        m.word_top = arrays_in['word_top']
        m.inv_top_sums = arrays_in['inv_top_sums']

    except (KeyError, TypeError):
        # Compatibility with old LDAGibbs class
        context_type = arrays_in['context_type'][()]
        K = arrays_in['K'][()]
        V = arrays_in['V'][()]
        alpha = arrays_in['alpha'][()]
        beta = arrays_in['beta'][()]
        
        m = ldaclass(context_type=context_type, K=K,
                     alpha=[alpha]*K, beta=[beta]*V)

        m.V = V
        m.corpus = arrays_in['W_corpus']

        m.Z_indices = arrays_in['Z_indices']
        m.Z_flat = arrays_in['Z_corpus']
        
        m.iteration = arrays_in['iterations'][()]
        m.log_probs = arrays_in['log_prob'].tolist()

        m.top_doc = arrays_in['doc_top'].T
        m.word_top = arrays_in['top_word'].T
        m.inv_top_sums = (1. / arrays_in['sum_word_top'])

    return m


def save_lda(m, filename):
    """
    Saves the model in an `.npz` file.
    
    :param filename: Name of file to be saved.
    :type filename: string
    
    :See Also: :class:`numpy.savez`
    """
    arrays_out = dict()
    
    arrays_out['K'] = m.K
    arrays_out['alpha'] = m.alpha
    arrays_out['beta'] = m.beta

    arrays_out['V'] = m.V
    arrays_out['context_type'] = m.context_type
    arrays_out['corpus'] = m.corpus

    arrays_out['Z_indices'] = m.Z_indices
    arrays_out['Z_flat'] = m.Z_flat

    arrays_out['iteration'] = m.iteration
    dt = dtype=[('i', np.int), ('v', np.float)]
    arrays_out['log_probs'] = np.array(m.log_probs, dtype=dt)

    arrays_out['top_doc'] = m.top_doc
    arrays_out['word_top'] = m.word_top
    arrays_out['inv_top_sums'] = m.inv_top_sums
    
    print 'Saving LDA model to', filename
    np.savez(filename, **arrays_out)
