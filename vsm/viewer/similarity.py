import numpy as np
from scipy.sparse import issparse

from vsm import enum_matrix, enum_sort, map_strarr, isstr, isint

from vsm.linalg import row_cosines, row_cos_mat, KL_divergence, JS_divergence, JS_dismat, posterior

from vsm.viewer import (
    res_word_type, res_doc_type, res_top_type, def_label_fn, doc_label_name)

from labeleddata import LabeledColumn, IndexedSymmArray

def_sim_fn = row_cosines
def_simmat_fn = row_cos_mat
def_order = 'd'

# TODO: Update module so that any function wrapping a similarity
# function assumes that similarity is computed row-wise

def sim_word_word(corp, mat, word_or_words, weights=None,
                  norms=None, as_strings=True, print_len=20,
                  filter_nan=True, sim_fn=row_cosines, order='d'):
    """
    Computes and sorts similarity of a word or list of words with 
    every other word. If weights are provided, the word list is
    represented as the weighted average of the words in the list. 
    Otherwise the arithmetic mean is used.

    Parameters
    ----------
    corp : Corpus
        Source of observed data
    mat : 2-dim floating point array
        Matrix based on which similarity is calculated
    word_or_words : string or list of string
        Query word(s) to which similarities are calculated
    weights : list of floating point
        Specify weights for each query word in `word_or_words`. 
        Default uses equal weights (i.e. arithmetic mean)
    norms : ?
        ?
    as_strings : boolean
        If true, returns a list of words rather than IDs. 
        Default is true.
    print_len : int
        Number of words printed by pretty-pringing function
        Default is 20.
    filter_nan : boolean
        ?

    """
    # Resolve `word_or_words`
    if isstr(word_or_words):
        word_or_words = [word_or_words]
    words, labels = zip(*[res_word_type(corp, w) for w in word_or_words])
    words, labels = list(words), list(labels)

    # Words are expected to be rows of `mat`

    # Generate pseudo-word
    if issparse(mat):
        rows = mat.tocsr()[words].toarray()
    else:
        rows = mat[words]
    word = np.average(rows, weights=weights, axis=0)[np.newaxis, :]
    
    # Compute similarities
    w_arr = sim_fn(word, mat, norms=norms)

    if as_strings:
        w_arr = enum_sort(w_arr, indices=corp.words, field_name='word',
			filter_nan=filter_nan)
    else:
    	w_arr = enum_sort(w_arr, filter_nan=filter_nan)

    if order=='d':
        pass
    elif order=='i':
        w_arr = w_arr[::-1]
    else:
        raise Exception('Invalid order parameter.')

    # Label data
    w_arr = w_arr.view(LabeledColumn)
    w_arr.col_header = 'Words: ' + ', '.join(labels)
    w_arr.subcol_headers = ['Word', 'Value']
    w_arr.col_len = print_len

    return w_arr


def sim_word_top(corp, mat, word_or_words, weights=[], norms=None,
                 print_len=10, filter_nan=True, sim_fn=KL_divergence, order='i'):
    """
    Computes (dis)similarity of a word or a list of words with every topic
    and sorts the results. The function treats the query words as a pseudo-topic
    that assign to those words non-zero probability masses specified by `weight`.
    Otherwise equal probability is assigned to each word in `word_or_words`. 
    """
    # Resolve `word_or_words`
    if isstr(word_or_words):
        word_or_words = [word_or_words]
    words, labels = zip(*[res_word_type(corp, w) for w in word_or_words])
    words, labels = list(words), list(labels)

    # Topics are assumed to be rows

    # Generate pseudo-topic
    top = np.zeros((1, mat.shape[1]), dtype=np.float64)
    if len(weights) == 0:
        top[:, words] = np.ones(len(words))
    else:
        top[:, words] = weights

    # Compute similarities
    k_arr = sim_fn(top, mat, norms=norms)
    k_arr = enum_sort(k_arr, filter_nan=filter_nan)

    if order=='d':
        pass
    elif order=='i':
        k_arr = k_arr[::-1]
    else:
        raise Exception('Invalid order parameter.')

    # Label data
    k_arr = k_arr.view(LabeledColumn)
    k_arr.col_header = 'Words: ' + ', '.join(labels)
    k_arr.subcol_headers = ['Topic', 'Value']
    k_arr.col_len = print_len

    return k_arr


def sim_top_doc(corp, mat, topic_or_topics, context_type, weights=[], 
                norms=None, print_len=10, filter_nan=True, 
                label_fn=def_label_fn, as_strings=True,
                sim_fn=row_cosines, order='d'):
    """
    Takes a topic or list of topics (by integer index) and returns a
    list of documents sorted by similarities or divergences calculated 
    according to `sim_fn`.
    """
    topics = res_top_type(topic_or_topics)
    # Assume documents are rows
            
    # Generate pseudo-document
    doc = np.zeros((1, mat.shape[1]), dtype=np.float64)
    if len(weights) == 0:
        doc[:, topics] = np.ones(len(topics))
    else:
        doc[:, topics] = weights
    # Compute similarites/divergences
    d_arr = sim_fn(doc, mat, norms=norms)

    # Label data
    if as_strings:
        md = corp.view_metadata(context_type)
        docs = label_fn(md)
        d_arr = enum_sort(d_arr, indices=docs, field_name='doc')
    else:
	d_arr = enum_sort(d_arr, filter_nan=filter_nan)

    if order=='d':
        pass
    elif order=='i':
        d_arr = d_arr[::-1]
    else:
        raise Exception('Invalid order parameter.')

    d_arr = d_arr.view(LabeledColumn)
    d_arr.col_header = 'Topics: ' + ', '.join([str(t) for t in topics])
    d_arr.subcol_headers = ['Document', 'Similarity']
    d_arr.col_len = print_len

    return d_arr



def sim_doc_doc(corp, mat, context_type, doc_or_docs, weights=None,
                norms=None, filter_nan=True, print_len=10,
                label_fn=def_label_fn, as_strings=True,
                sim_fn=KL_divergence, order='i'):
    """
    Computes similarities of a document (or a list of documents) 
    to every other documents in the model and sorts the result.
    """
    # Resolve `doc_or_docs`
    label_name = doc_label_name(context_type)    
    if (isstr(doc_or_docs) or isint(doc_or_docs) 
        or isinstance(doc_or_docs, dict)):
        docs = [res_doc_type(corp, context_type, label_name, doc_or_docs)[0]]
    else:
        docs = [res_doc_type(corp, context_type, label_name, d)[0] 
                for d in doc_or_docs]

    # Assume documents are columns, so transpose
    mat = mat.T

    # Generate pseudo-document
    if issparse(mat):
        rows = mat.tocsr()[docs].toarray()
    else:
        rows = mat[docs]
    doc = np.average(rows, weights=weights, axis=0)[np.newaxis, :]

    # Compute (dis)similarities
    d_arr = sim_fn(doc, mat, norms=norms)

    # Label data
    if as_strings:
        md = corp.view_metadata(context_type)
        docs = label_fn(md)
        d_arr = enum_sort(d_arr, indices=docs, field_name='doc')
    else:
        d_arr = enum_sort(d_arr, filter_nan=filter_nan)

    if order=='d':
        pass
    elif order=='i':
        d_arr = d_arr[::-1]
    else:
        raise Exception('Invalid order parameter.')

    d_arr = d_arr.view(LabeledColumn)
    # TODO: Finish this header
    d_arr.col_header = 'Documents: '
    d_arr.subcol_headers = ['Document', 'Value']
    d_arr.col_len = print_len

    return d_arr


def sim_top_top(mat, topic_or_topics, weights=None, 
                norms=None, print_len=10,
                filter_nan=True, sim_fn=KL_divergence, order='i'):
    """
    Computes similarities of a topic (or a list of topics) to 
    every other topics and sorts the result.
    """
    topics = res_top_type(topic_or_topics)

    # Assuming topics are rows

    # Generate pseudo-topic
    top = np.average(mat[topics], weights=weights, axis=0)[np.newaxis, :]

    # Compute similarities
    k_arr = sim_fn(top, mat, norms=norms)
    k_arr = enum_sort(k_arr, filter_nan=filter_nan)

    if order=='d':
        pass
    elif order=='i':
        k_arr = k_arr[::-1]
    else:
        raise Exception('Invalid order parameter.')

    # Label data
    k_arr = k_arr.view(LabeledColumn)
    k_arr.col_header = 'Topics: ' + ', '.join([str(t) for t in topics])
    k_arr.subcol_headers = ['Topic', 'Value']
    k_arr.col_len = print_len

    return k_arr



def simmat_words(corp, matrix, word_list, norms=None, sim_fn=row_cos_mat):
    """
    """
    indices, words = zip(*[res_word_type(corp, word) 
                           for word in word_list])
    indices, words = np.array(indices), np.array(words)

    sm = sim_fn(indices, matrix, norms=norms, fill_tril=True)
    sm = sm.view(IndexedSymmArray)
    sm.labels = words
    
    return sm



def simmat_documents(corp, matrix, context_type, doc_list,
                     norms=None, sim_fn=JS_dismat):
    """
    If sim_fn=JS_dismat, output is distance matirx.
    If sim_fn=row_cos_mat, output is similarity matrix.
    """

    label_name = doc_label_name(context_type)

    indices, labels = zip(*[res_doc_type(corp, context_type, label_name, doc) 
                            for doc in doc_list])
    indices, labels = np.array(indices), np.array(labels)

    sm = sim_fn(indices, matrix.T, norms=norms, fill_tril=True)
    sm = sm.view(IndexedSymmArray)
    sm.labels = labels
    
    return sm



def simmat_topics(kw_mat, topics, norms=None, sim_fn=JS_dismat):
    """
    If sim_fn=JS_dismat, output is distance matirx.
    If sim_fn=row_cos_mat, output is similarity matrix.
    """
    sm = sim_fn(topics, kw_mat, norms=norms, fill_tril=True)
    sm = sm.view(IndexedSymmArray)
    sm.labels = [str(k) for k in topics]
    
    return sm
