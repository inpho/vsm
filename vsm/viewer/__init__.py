import numpy as np

from vsm import (enum_sort, row_normalize, map_strarr, 
                 isfloat, isint, isstr)
from vsm import corpus as cps
from vsm.util.corpustools import word_tokenize
from vsm import model

from similarity import row_cosines, simmat_rows



class Viewer(object):
    """
    """
    def __init__(self,
                 corpus=None,
                 matrix=None,
                 tok_name=None):

        self.corpus = corpus
        self.matrix = matrix
        self.tok_name = tok_name


    def load_matrix(self, filename):

        self.matrix = model.Model.load_matrix(filename)


    def load_corpus(self, filename):

        self.corpus = cps.Corpus.load(filename)



def simmat_terms(corpus, matrix, term_list):
    """
    """
    indices, terms = zip([res_term_type(corpus, term) 
                          for term in term_list])

    simmat = simmat_rows(matrix, indices)

    simmat.labels = terms
    
    return simmat



def simmat_documents(corpus, matrix, tok_name, doc_list):
    """
    """
    indices, labels = zip([res_doc_type(corpus, tok_name, doc) 
                           for doc in doc_list])

    simmat = simmat_rows(matrix.T, indices)

    simmat.labels = labels

    return simmat



def simmat_topics(kw_mat, topics):
    """
    """
    simmat = simmat_rows(kw_mat, topics)

    simmat.labels = [str(k) for k in topics]

    return simmat



def similar_terms(corpus, matrix, term,
                  norms=None, filter_nan=True,
                  rem_masked=True):
    """
    """
    i, term = res_term_type(corpus, term)
    
    t_arr = row_cosines(matrix[i:i+1, :], matrix, 
                        norms=norms, filter_nan=filter_nan)

    t_arr = np.array([(corpus.terms[i], v) for i,v in t_arr],
                        dtype=[('i', corpus.terms.dtype),
                               ('value', t_arr.dtype['value'])])

    if rem_masked:

        f = np.vectorize(lambda x: x is not np.ma.masked)

        t_arr = t_arr[f(t_arr['i'])]

    t_arr = t_arr.view(IndexedValueArray)

    t_arr.main_header = term

    t_arr.subheaders = [('Term', 'Cosine')]

    return t_arr



def similar_documents(corpus, matrix, tok_name, doc,
                      norms=None, filter_nan=True):
    """
    """
    label_name = doc_label_name(tok_name)

    d, label = res_doc_type(corpus, tok_name, label_name, doc)

    d_arr = row_cosines(matrix[:, d:d+1].T, matrix.T, norms=norms,
                        filter_nan=filter_nan)

    docs = corpus.view_metadata(tok_name)[label_name]

    d_arr = np.array([(docs[i], v) for i,v in d_arr],
                     dtype=[('doc', docs.dtype), 
                            ('value', d_arr.dtype['value'])])

    d_arr = d_arr.view(IndexedValueArray)

    d_arr.main_header = label

    d_arr.subheaders = [('Document', 'Cosine')]

    return d_arr



def mean_similar_terms(corpus, matrix, query,
                       norms=None, filter_nan=True,
                       rem_masked=True):
    """
    """
    terms = word_tokenize(query)

    t_arr = []

    for term in terms:

        t, term = res_term_type(corpus, term)

        cosines = row_cosines(matrix[t:t+1, :], matrix, norms=norms,
                              sort=False, filter_nan=False)['value']
        
        t_arr.append(cosines)

    t_arr = np.mean(t_arr, axis=0)

    t_arr = enum_sort(t_arr)
    
    if filter_nan:

        t_arr = t_arr[np.isfinite(t_arr['value'])]

    t_arr = np.array([(corpus.terms[i], v) for i,v in t_arr],
                     dtype=[('i', corpus.terms.dtype),
                            ('value', t_arr.dtype['value'])])
    
    if rem_masked:

        f = np.vectorize(lambda x: x is not np.ma.masked)
        
        t_arr = t_arr[f(t_arr['i'])]
        
    t_arr = t_arr.view(IndexedValueArray)

    t_arr.main_header = query

    t_arr.subheaders = [('Term', 'Cosine')]

    return t_arr



def sim_top_top(kw_mat, k, norms=None, filter_nan=True):
    """
    Computes and sorts the cosine values between a given topic `k`
    and every topic.
    """
    k_arr = row_cosines(kw_mat[k:k+1, :], kw_mat, 
                        norms=norms, filter_nan=filter_nan)

    k_arr = k_arr.view(IndexedValueArray)

    k_arr.main_header = 'Topic ' + str(k)

    k_arr.subheaders = [('Topic', 'Cosine')]

    return k_arr



def sim_word_avg_top(corpus, wk_mat, words, weights=None,
                     norms=None, filter_nan=True):
    """
    Computes and sorts the cosine values between a list of words
    `words` and every topic. If weights are not provided, the word
    list is represented in the space of topics as a topic which
    assigns equal probability to each word in `words` and 0 to
    every other word in the corpus. Otherwise, each word in
    `words` is assigned the provided weight.
    """
    col = np.zeros((wk_mat.shape[0], 1), dtype=np.float64)

    for i in xrange(len(words)):

        w, word = res_term_type(corpus, words[i])

        if weights:
            
            col[w, 0] = weights[i]
            
        else:

            col[w, 0] = 1

    k_arr = row_cosines(col.T, wk_mat.T, norms=norms, filter_nan=filter_nan)

    k_arr = k_arr.view(IndexedValueArray)

    k_arr.main_header = 'Words: ' + ', '.join(words)

    k_arr.subheaders = [('Topic', 'Cosine')]

    return k_arr



def sim_word_avg_word(corpus, mat, words, weights=None, norms=None,
                      as_strings=True, filter_nan=True):
    """
    Computes and sorts the cosine values between a list of words
    `words` and every word. If weights are provided, the word list
    is represented as the weighted average of the words in the
    list. If weights are not provided, the arithmetic mean is
    used.
    """
    # Words are expected to be rows of `mat`
    row = np.zeros((1, mat.shape[1]), dtype=np.float64)

    for i in xrange(len(words)):

        w, word = res_term_type(corpus, words[i])

        if weights:
            
            row[0, :] += mat[w, :] * weights[i]
            
        else:

            row[0, :] += mat[w, :]

    w_arr = row_cosines(row, mat, norms=norms, filter_nan=filter_nan)

    if as_strings:
        
        w_arr = map_strarr(w_arr, corpus.terms, k='i')

    w_arr = w_arr.view(IndexedValueArray)

    w_arr.main_header = 'Words: ' + ', '.join(words)

    w_arr.subheaders = [('Word', 'Cosine')]

    return w_arr



def format_entry(x):

    # np.void is the type of the tuples that appear in numpy
    # structured arrays
    if isinstance(x, np.void):

        return ', '.join([format_entry(i) for i in x.tolist()]) 
        
    if isfloat(x):

        return '{0:.5f}'.format(x)

    return str(x)


        
class IndexedValueArray(np.ndarray):
    """
    """
    def __new__(cls, input_array, main_header=None, subheaders=None):

        obj = np.asarray(input_array).view(cls)

        obj.str_len = None

        obj.main_header = main_header

        obj.subheaders = subheaders

        return obj



    def __array_finalize__(self, obj):

        if obj is None: return

        self.str_len = getattr(obj, 'str_len', None)

        self.main_header = getattr(obj, 'main_header', None)

        self.subheaders = getattr(obj, 'subheaders', None)



    def __str__(self):

        if self.ndim == 1:

            arr = self[np.newaxis, :]

        elif self.ndim == 2:

            arr = self

        else:

            return super(IndexedValueArray, self).__str__()

        vsep_1col = '-' * 37 + '\n'

        vsep_2col = '-' * 75 + '\n'

        if arr.main_header:
        
            if arr.shape[0] == 1:
                
                s = vsep_1col

                s += '{0:^35}\n'.format(arr.main_header)

            else:

                s = vsep_2col

                s += '{0:^75}\n'.format(arr.main_header)

        else:

            s = ''

        m = arr.shape[0]

        if self.str_len:

            n = min(arr.shape[1], self.str_len)

        else:
            
            n = arr.shape[1]

        for i in xrange(0, m - m % 2, 2):

            if arr.subheaders:

                s += vsep_2col

                s += ('{0:<25}{1:<15}{2:<25}{3}\n'
                      .format(arr.subheaders[i][0], 
                              arr.subheaders[i][1],
                              arr.subheaders[i+1][0], 
                              arr.subheaders[i+1][1]))
                                      
            s += vsep_2col

            for j in xrange(n):

                a0 = format_entry(arr[i][j][0])

                a1 = format_entry(arr[i][j][1])

                b0 = format_entry(arr[i+1][j][0])

                b1 = format_entry(arr[i+1][j][1])

                s += '{0:<25}{1:<15}{2:<25}{3}\n'.format(a0, a1, b0, b1)

        if m % 2:

            if arr.subheaders:

                s += vsep_1col

                s += ('{0:<25}{1}\n'
                      .format(arr.subheaders[m-1][0], 
                              arr.subheaders[m-1][1]))
                                      
            s += vsep_1col

            for j in xrange(n):

                a0 = format_entry(arr[m-1][j][0])

                a1 = format_entry(arr[m-1][j][1])

                s += '{0:<25}{1}\n'.format(a0, a1)
            
        return s



def doc_label_name(tok_name):

    return tok_name + '_label'



def res_doc_type(corpus, tok_name, label_name, doc):
    """
    If `doc` is a string or a dict, performs a look up for its
    associated integer. If `doc` is a dict, looks for its label.
    Finally, if `doc` is an integer, stringifies `doc` for use as
    a label. 
    
    Returns an integer, string pair: (<document index>, <document
    label>).
    """
    if isinstance(doc, basestring):
        
        query = {label_name: doc}
        
        d = corpus.meta_int(tok_name, query)
        
    elif isinstance(doc, dict):
        
        d = corpus.meta_int(tok_name, doc)
        
        #TODO: Define an exception for failed queries in
        #vsm.corpus. Use it here.
        
        doc = corpus.view_metadata(tok_name)[label_name][d]

    else:

        d, doc = doc, str(doc)

    return d, doc
                    
            

def res_term_type(corpus, term):
    """
    If `term` is a string, performs a look up for its associated
    integer. Otherwise, stringifies `term`. 

    Returns an integer, string pair: (<term index>, <term label>).
    """
    if isinstance(term, basestring):
            
        return corpus.terms_int[term], term

    return term, str(term)
        


############################################################
#                        Testing
############################################################

def test_IndexedValueArray():

    terms = ['row', 'row', 'row', 'your', 'boat', 'gently', 'down', 'the', 
             'stream', 'merrily', 'merrily', 'merrily', 'merrily', 'life', 
             'is', 'but', 'a', 'dream']

    values = [np.random.random() for t in terms]

    d = [('i', np.array(terms).dtype), 
         ('value', np.array(values).dtype)]

    v = np.array(zip(terms, values), dtype=d)

    arr = np.vstack([v] * 5)

    arr = arr.view(IndexedValueArray)

    arr.main_header = 'Test 2-d Array'

    arr.subheaders = [('Repetition ' + str(i), 'Random') 
                      for i in xrange(arr.shape[0])]
    
    print arr

    print

    arr = v.view(IndexedValueArray)

    arr.main_header = 'Test 1-d Array'
    
    print arr
