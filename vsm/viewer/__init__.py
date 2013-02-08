import numpy as np

from vsm import (enum_sort, row_normalize, map_strarr, 
                 isfloat, isint, isstr)
from vsm import corpus as cps
from vsm.util.corpustools import word_tokenize
from vsm import model

from similarity import row_cosines, row_cos_mat


# TODO: Update module so that any function wrapping a similarity
# function assumes that similarity is computed row-wise

def sim_word_word(corp, mat, word_or_words, weights=None, norms=None,
                  as_strings=True, print_len=20, filter_nan=True):
    """
    Computes and sorts the cosine values between a word or list of
    words and every word. If weights are provided, the word list is
    represented as the weighted average of the words in the list. If
    weights are not provided, the arithmetic mean is used.
    """
    # Resolve `word_or_words`
    if isstr(word_or_words):
        word_or_words = [word_or_words]
    words, labels = zip(*[res_term_type(corp, w) for w in word_or_words])
    words, labels = list(words), list(labels)

    # Words are expected to be rows of `mat`

    # Generate pseudo-word
    word = np.average(mat[words], weights=weights, axis=0)[np.newaxis, :]
    
    # Compute similarities
    w_arr = row_cosines(word, mat, norms=norms, filter_nan=filter_nan)

    # Label data
    if as_strings:
        w_arr = map_strarr(w_arr, corp.terms, k='i', new_k='word')
    w_arr = w_arr.view(IndexedValueArray)
    w_arr.main_header = 'Words: ' + ', '.join(labels)
    w_arr.subheaders = [('Word', 'Cosine')]
    w_arr.str_len = print_len

    return w_arr


def sim_word_top(corp, mat, word_or_words, weights=[],
                 norms=None, print_len=10, filter_nan=True):
    """
    Computes and sorts the cosine values between a word or a list of
    words and every topic. If weights are not provided, the word list
    is represented in the space of topics as a topic which assigns
    equal non-zero probability to each word in `words` and 0 to every
    other word in the corpus. Otherwise, each word in `words` is
    assigned the provided weight.
    """
    # Resolve `word_or_words`
    if isstr(word_or_words):
        word_or_words = [word_or_words]
    words, labels = zip(*[res_term_type(corp, w) for w in word_or_words])
    words, labels = list(words), list(labels)

    # Topics are assumed to be rows

    # Generate pseudo-topic
    top = np.zeros((1, mat.shape[1]), dtype=np.float64)
    if len(weights) == 0:
        top[:, words] = np.ones(len(words))
    else:
        top[:, words] = weights

    # Compute similarities
    k_arr = row_cosines(top, mat, norms=norms, filter_nan=filter_nan)

    # Label data
    k_arr = k_arr.view(IndexedValueArray)
    k_arr.main_header = 'Words: ' + ', '.join(labels)
    k_arr.subheaders = [('Topic', 'Cosine')]
    k_arr.str_len = print_len

    return k_arr


def sim_top_doc(corp, mat, topic_or_topics, tok_name, weights=[], 
                norms=None, print_len=10, as_strings=True, filter_nan=True):
    """
    Takes a topic or list of topics (by integer index) and returns a
    list of documents sorted by the posterior probabilities of
    documents given the topic.
    """
    if isint(topic_or_topics):
        topic_or_topics = [topic_or_topics]
    topics = topic_or_topics
            
    # Assume documents are rows

    # Generate pseudo-document
    doc = np.zeros((1, mat.shape[1]), dtype=np.float64)
    if len(weights) == 0:
        doc[:, topics] = np.ones(len(topics))
    else:
        doc[:, topics] = weights

    # Compute similarites
    d_arr = row_cosines(doc, mat, norms=norms, filter_nan=filter_nan)

    # Label data
    if as_strings:
        md = corp.view_metadata(tok_name)
        docs = md[doc_label_name(tok_name)]
        d_arr = map_strarr(d_arr, docs, k='i', new_k='doc')
        
    d_arr = d_arr.view(IndexedValueArray)
    d_arr.main_header = 'Topics: ' + ', '.join([str(t) for t in topics])
    d_arr.subheaders = [('Document', 'Prob')]
    d_arr.str_len = print_len

    return d_arr


# Add as_strings parameter
def sim_doc_doc(corp, mat, tok_name, doc_or_docs, weights=None,
                norms=None, filter_nan=True, print_len=10, as_strings=True):
    """
    """
    # Resolve `word_or_words`
    label_name = doc_label_name(tok_name)    
    if (isstr(doc_or_docs) or isint(doc_or_docs) 
        or isinstance(doc_or_docs, dict)):
        docs = [res_doc_type(corp, tok_name, label_name, doc_or_docs)[0]]
    else:
        docs = [res_doc_type(corp, tok_name, label_name, d)[0] 
                for d in doc_or_docs]

    # Assume documents are columns, so transpose
    mat = mat.T

    # Generate pseudo-document
    doc = np.average(mat[docs], weights=weights, axis=0)[np.newaxis, :]

    # Compute cosines
    d_arr = row_cosines(doc, mat, norms=norms, filter_nan=filter_nan)

    # Label data
    if as_strings:
        docs = corp.view_metadata(tok_name)[label_name]
        d_arr = map_strarr(d_arr, docs, k='i', new_k='doc')
    
    d_arr = d_arr.view(IndexedValueArray)
    # TODO: Finish this header
    d_arr.main_header = 'Documents: '
    d_arr.subheaders = [('Document', 'Cosine')]
    d_arr.str_len = print_len
    
    return d_arr


def sim_top_top(mat, topic_or_topics, weights=None, 
                norms=None, print_len=10, filter_nan=True):
    """
    Computes and sorts the cosine values between a given topic `k`
    and every topic.
    """
    # Resolve `topic_or_topics`
    if isint(topic_or_topics):
        topic_or_topics = [topic_or_topics]
    topics = topic_or_topics

    # Assuming topics are rows

    # Generate pseudo-topic
    top = np.average(mat[topics], weights=weights, axis=0)[np.newaxis, :]

    # Compute similarities
    k_arr = row_cosines(top, mat, norms=norms, filter_nan=filter_nan)

    # Label data
    k_arr = k_arr.view(IndexedValueArray)
    k_arr.main_header = 'Topics: ' + ', '.join([str(t) for t in topics])
    k_arr.subheaders = [('Topic', 'Cosine')]
    k_arr.str_len = print_len

    return k_arr


# TODO: Update module and remove this wrapper
def similar_documents(corp, matrix, tok_name, doc,
                      norms=None, filter_nan=True):

    return sim_doc_doc(corp, matrix, tok_name, doc,
                       norms=norms, filter_nan=filter_nan)


# TODO: Update module and remove this wrapper
def similar_terms(corp, matrix, term,
                  norms=None, filter_nan=True,
                  rem_masked=True):
    """
    """
    return sim_word_word(corp, matrix, [term],
                         norms=norms, filter_nan=True)


# TODO: Update module and remove this wrapper
def mean_similar_terms(corp, matrix, query,
                       norms=None, filter_nan=True,
                       rem_masked=True):
    """
    """
    words = word_tokenize(query)

    return sim_word_word(corp, matrix, words,
                         norms=norms, filter_nan=True)
    

def simmat_terms(corp, matrix, term_list, norms=None):
    """
    """
    indices, terms = zip(*[res_term_type(corp, term) 
                           for term in term_list])

    indices, terms = np.array(indices), np.array(terms)

    sm = row_cos_mat(indices, matrix, norms=norms, fill_tril=True)

    sm = sm.view(IndexedSymmArray)

    sm.labels = terms
    
    return sm



def simmat_documents(corp, matrix, tok_name, doc_list, norms=None):
    """
    """
    label_name = doc_label_name(tok_name)

    indices, labels = zip(*[res_doc_type(corp, tok_name, label_name, doc) 
                            for doc in doc_list])

    indices, labels = np.array(indices), np.array(labels)

    sm = row_cos_mat(indices, matrix.T, norms=norms, fill_tril=True)

    sm = sm.view(IndexedSymmArray)

    sm.labels = labels
    
    return sm



def simmat_topics(kw_mat, topics, norms=None):
    """
    """
    sm = row_cos_mat(topics, kw_mat, norms=norms, fill_tril=True)

    sm = sm.view(IndexedSymmArray)

    sm.labels = [str(k) for k in topics]
    
    return sm


# TODO: Investigate compressed forms of symmetric matrix. Cf.
# scipy.spatial.distance.squareform
class IndexedSymmArray(np.ndarray):
    """
    """
    def __new__(cls, input_array, labels=None):

        obj = np.asarray(input_array).view(cls)

        obj.labels = labels

        return obj


    def __array_finalize__(self, obj):

        if obj is None: return

        self.labels = getattr(obj, 'labels', None)



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



def res_doc_type(corp, tok_name, label_name, doc):
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
        
        d = corp.meta_int(tok_name, query)
        
    elif isinstance(doc, dict):
        
        d = corp.meta_int(tok_name, doc)
        
        #TODO: Define an exception for failed queries in
        #vsm.corpus. Use it here.
        
        doc = corp.view_metadata(tok_name)[label_name][d]

    else:

        d, doc = doc, str(doc)

    return d, doc
                    
            

def res_term_type(corp, term):
    """
    If `term` is a string, performs a look up for its associated
    integer. Otherwise, stringifies `term`. 

    Returns an integer, string pair: (<term index>, <term label>).
    """
    if isinstance(term, basestring):
            
        return corp.terms_int[term], term

    return term, str(term)
        

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
