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


def categorical(pvals, random_seed=0):
    """
    Draws a sample from the categorical distribution parameterized by
    `pvals`.
    """
    random_state = np.random.RandomState(random_seed)
    cum_dist = np.cumsum(pvals)
    r = random_state.uniform() * cum_dist[-1]
    return np.searchsorted(cum_dist, r)


def cgs_update(itr, docs, word_top, inv_top_sums, 
               top_doc, Z, indices=None, random_seed=0):

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

            k = categorical(dist, random_seed=random_seed)

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

    context_type = arrays_in['context_type'][()]
    K = arrays_in['K'][()]

    if 'm_words' in arrays_in:
        V = arrays_in['m_words'][()]
    else:
        V = arrays_in['V'][()]

    if 'ctx_prior' in arrays_in:
        alpha = arrays_in['ctx_prior']
    elif arrays_in['alpha'].size==1:
        alpha = np.ones(K, dtype=np.float64) * arrays_in['alpha'][()]
    else:
        alpha = arrays_in['alpha']

    if 'top_prior' in arrays_in:
        beta = arrays_in['top_prior']
    elif arrays_in['beta'].size==1:
        beta = np.ones(V, dtype=np.float64) * arrays_in['beta'][()]
    else:
        beta = arrays_in['beta']
        
    m = ldaclass(context_type=context_type, K=K, V=V, alpha=alpha, beta=beta)

    if 'W_indices' in arrays_in:
        m.indices = arrays_in['W_indices']
    elif 'contexts' in arrays_in: 
        m.indices = [s.stop for s in arrays_in['contexts']]
    else:
        m.indices = arrays_in['indices']

    if 'W_corpus' in arrays_in:
        m.corpus = arrays_in['W_corpus']
    else:
        m.corpus = arrays_in['corpus']        

    if 'Z_corpus' in arrays_in:
        m.Z = arrays_in['Z_corpus']        
    else:
        m.Z = arrays_in['Z']

    if 'top_word' in arrays_in:
        m.word_top = arrays_in['top_word'].T
    else:
        m.word_top = arrays_in['word_top']

    if 'doc_top' in arrays_in:
        m.top_doc = arrays_in['doc_top'].T
    elif 'top_ctx' in arrays_in:
        m.top_doc = arrays_in['top_ctx']
    else:
        m.top_doc = arrays_in['top_doc']

    if 'sum_word_top' in arrays_in:
        m.inv_top_sums = 1. / arrays_in['sum_word_top']
    elif 'top_norms' in arrays_in:
        m.inv_top_sums = arrays_in['top_norms']
    else:
        m.inv_top_sums = arrays_in['inv_top_sums']

    if 'iterations' in arrays_in:
        m.iteration = arrays_in['iterations'][()]
    else:
        m.iteration = arrays_in['iteration'][()]

    if 'log_prob' in arrays_in:
        m.log_probs = arrays_in['log_prob'].tolist()
    else:
        m.log_probs = arrays_in['log_probs'].tolist()

    return m


def save_lda(m, filename):
    """
    Saves the model in an `.npz` file.
    
    :param filename: Name of file to be saved.
    :type filename: string
    
    :See Also: :class:`numpy.savez`
    """
    arrays_out = dict()

    arrays_out['context_type'] = m.context_type

    arrays_out['alpha'] = m.alpha
    arrays_out['beta'] = m.beta

    arrays_out['K'] = m.K
    arrays_out['V'] = m.V
    arrays_out['indices'] = m.indices
    arrays_out['corpus'] = m.corpus
    arrays_out['Z'] = m.Z

    arrays_out['iteration'] = m.iteration
    dt = dtype=[('i', np.int), ('v', np.float)]
    arrays_out['log_probs'] = np.array(m.log_probs, dtype=dt)

    arrays_out['top_doc'] = m.top_doc
    arrays_out['word_top'] = m.word_top
    arrays_out['inv_top_sums'] = m.inv_top_sums
    
    print 'Saving LDA model to', filename
    np.savez(filename, **arrays_out)
