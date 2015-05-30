import numpy as np


__all__ = [ 'init_priors', 'compute_top_doc', 'compute_word_top', 
            'compute_log_prob', 'load_lda', 'save_lda' ]



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

    if 'seed' in arrays_in:
        m.seed = int(arrays_in['seed'])
        m._mtrand_state = (str(arrays_in['mtrand_state0']),
                           arrays_in['mtrand_state1'],
                           int(arrays_in['mtrand_state2']),
                           int(arrays_in['mtrand_state3']),
                           float(arrays_in['mtrand_state4']))

    if 'seeds' in arrays_in:
        m.seeds = map(int, list(arrays_in['seeds']))
        m._mtrand_states = zip(map(str, arrays_in['mtrand_states0']),
                               arrays_in['mtrand_states1'],
                               map(int, arrays_in['mtrand_states2']),
                               map(int, arrays_in['mtrand_states3']),
                               map(float, arrays_in['mtrand_states4']))
        m.n_proc = len(m.seeds)

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

    if hasattr(m,'seed'):
        arrays_out['seed'] = m.seed
    if hasattr(m,'seeds'):
        arrays_out['seeds'] = m.seeds
    if hasattr(m, '_mtrand_state'):
        for i,s in enumerate(m._mtrand_state):
            key = 'mtrand_state{0}'.format(i)
            arrays_out[key] = s
    
    if hasattr(m, '_mtrand_states'):
        for i,s in enumerate(zip(*m._mtrand_states)):
            key = 'mtrand_states{0}'.format(i)
            arrays_out[key] = s
    
    print 'Saving LDA model to', filename
    np.savez(filename, **arrays_out)


def compute_log_prob(W, Z, word_top, top_doc):
    log_wt = np.log(word_top / word_top.sum(0))
    log_td = np.log(top_doc / top_doc.sum(0))
    log_prob = 0
    for i in xrange(len(W)):
        W_i, Z_i = W[i], Z[i]
        for j in xrange(len(W_i)):
            w, k = W_i[j], Z_i[j]
            log_prob += log_wt[w,k] + log_td[k,i]
    return log_prob


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
