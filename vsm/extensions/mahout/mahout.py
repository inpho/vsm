import re
import numpy as np


def load_help(txtfile):
    """
    Returns a list of strings split with ': '
    """
    with open(txtfile, 'r') as f:
        s = f.read()
        s = re.sub('\n',': ', s)
        li = s.split(': ')
        return li


def load_vals(txtfile):
    """
    Loads data from mahout-generated txtfile(topic-term or doc-topic).
    Returns a list of dictionaries.
    """
    import ast
    
    data = []
    li = load_help(txtfile)
        
    for i in xrange(len(li)):
        if li[i] == 'Value' and i < len(li)-1:
            dic = ast.literal_eval(li[i+1])
            data.append(dic)
    return data


def build_arr(dictli):
    """
    dictli : list of dictionaries
    """
    r = len(dictli)
    c = len(dictli[0])
    
    arr = np.zeros((r,c))
    
    for i in xrange(r):
        arr[i] = dictli[i].values()
        
    return arr


def load_kv(txtfile):
    """
    Returns dictionary equivalent to Corpus.word_int
    """
    dic = {}
    li = load_help(txtfile)
        
    for i in xrange(len(li)):
        if li[i] == 'Key':
            dic[li[i+1]] = int(li[i+3])
            
    return dic


def make_corpus(txtfile, word_int, as_strings=False):
    """
    Returns a list of arrays that represent documents.
    """
    corp = []
    li = load_help(txtfile)
    
    for i in xrange(len(li)):
        if li[i] == 'Value':
            doc = li[i+1]
            doc = doc.strip()
            doc = doc.strip('[')
            doc = doc.strip(']')
            doc = doc.split(', ')
            doc = [str(w) for w in doc]
            
            idoc = []
            for w in doc:
                try:
                    i = word_int[w]
                    if as_strings:
                        idoc.append(w)
                    else:
                        idoc.append(int(i))
                except:
                    pass
            
            corp.append(np.array(idoc))
            
    return corp


def stopwords(corp, topword):
    """
    corp : `Corpus` object
    topword : topword (list of dictionaries) from model.
    """
    ind = topword[0].keys()

    rem = []
    for w in corp.words:
        i = corp.words_int[w]
        if i not in ind:
            rem.append(w)

    return rem


def savez(fname, ctx_type, itr, K, alpha, beta, doc_top, top_word, W):
    arrays_out = dict()
    
    V = top_word.shape[1]
    # use mahout-vect-test/tokenized-documents
    # mahout-vect-test/dictionary.file-0
    corp = np.array(np.hstack(W))
    arrays_out['W_corpus'] = corp 
    arrays_out['W_indices'] = np.cumsum([a.size for a in W])
    arrays_out['V'] = V # num of Vocabs
    
    # next 3 lines are dummy values
    arrays_out['Z_corpus'] = np.zeros(corp.shape[0])
    arrays_out['Z_indices'] = np.cumsum([a.size for a in W])
    arrays_out['log_prob_init'] = False
    
    arrays_out['doc_top'] = doc_top
    arrays_out['top_word'] = top_word
    arrays_out['sum_word_top'] = (V * beta) + np.zeros(K)
                                  
    arrays_out['context_type'] = ctx_type
    arrays_out['K'] = K
    arrays_out['iterations'] = itr
    arrays_out['alpha'] = alpha
    arrays_out['beta'] = beta
    
    print 'Saving LDA model to ', fname
    np.savez(fname, **arrays_out)


"""
if __name__=='__main__':
    # workflow
    # from vsm.corpus.util.corpupsbuilders import corpus_fromlist

    # Return topword, doctop information from the txt file as arrays.
    top_word = load_vals('../../../mahout-lda-test/lda.txt')
    doc_top = load_vals('../../../mahout-dt-test/doc-topics.txt')

    arrtw = build_arr(top_word)
    arrdt = build_arr(doc_top)
    
    # dicionary that corresponds to Corpus.words_int
    words_int = load_kv('../../../mahout-vect-test/dict.txt')

    # list of arrays that represent documents.
    # `wcorp` can be an input to `corpus_fromlist()` to create a `Corpus`.
    wcorp = make_corpus('../../../mahout-vect-test/tokenized-documents/tdocs.txt',
                        words_int, as_strings=True)

    # make `Corpus` object and apply_stoplist to ensure the words
    # are exactly the same as the ones in the topword.
    wc = corpus_fromlist(wcorp, 'document')
    rem = stopwords(wc, top_word)
    wc_ = wc.apply_stoplist(rem)

    # Save `Corpus` and LDA model.
    wc_.save('mahout-test.npz')
    savez('mahout-test-K5-100.npz', 'document', 100, 5, 0.01, 0.01, arrdt,
            arrtw, wc_.view_contexts('document'))

""" 
