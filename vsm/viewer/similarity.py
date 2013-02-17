import numpy as np
from scipy.sparse import issparse

from vsm import enum_sort, map_strarr, isstr, isint

from vsm.linalg import row_cosines, row_cos_mat

from vsm.viewer import (
    res_word_type, res_doc_type, def_label_fn, doc_label_name)

from labeleddata import (
    LabeledColumn, IndexedValueArray, IndexedSymmArray)


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
    w_arr = row_cosines(word, mat, norms=norms)
    w_arr = enum_sort(w_arr, filter_nan=True)

    # Label data
    if as_strings:
        w_arr = map_strarr(w_arr, corp.words, k='i', new_k='word')
    w_arr = w_arr.view(LabeledColumn)
    w_arr.col_header = 'Words: ' + ', '.join(labels)
    w_arr.subcol_headers = ['Word', 'Cosine']
    w_arr.col_len = print_len

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
    k_arr = row_cosines(top, mat, norms=norms)
    k_arr = enum_sort(k_arr, filter_nan=filter_nan)

    # Label data
    k_arr = k_arr.view(LabeledColumn)
    k_arr.col_header = 'Words: ' + ', '.join(labels)
    k_arr.subcol_headers = ['Topic', 'Cosine']
    k_arr.col_len = print_len

    return k_arr


def sim_top_doc(corp, mat, topic_or_topics, context_type, weights=[], 
                norms=None, print_len=10, filter_nan=True, 
                label_fn=def_label_fn, as_strings=True):
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
    d_arr = row_cosines(doc, mat, norms=norms)
    d_arr = enum_sort(d_arr, filter_nan=filter_nan)

    # Label data
    if as_strings:
        md = corp.view_metadata(context_type)
        docs = label_fn(md)
        d_arr = map_strarr(d_arr, docs, k='i', new_k='doc')
    d_arr = d_arr.view(LabeledColumn)
    d_arr.col_header = 'Topics: ' + ', '.join([str(t) for t in topics])
    d_arr.subcol_headers = ['Document', 'Prob']
    d_arr.col_len = print_len

    return d_arr



def sim_doc_doc(corp, mat, context_type, doc_or_docs, weights=None,
                norms=None, filter_nan=True, print_len=10, 
                label_fn=def_label_fn, as_strings=True):
    """
    """
    # Resolve `word_or_words`
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

    # Compute cosines
    d_arr = row_cosines(doc, mat, norms=norms)
    d_arr = enum_sort(d_arr, filter_nan=filter_nan)

    # Label data
    if as_strings:
        md = corp.view_metadata(context_type)
        docs = label_fn(md)
        d_arr = map_strarr(d_arr, docs, k='i', new_k='doc')
    d_arr = d_arr.view(LabeledColumn)
    # TODO: Finish this header
    d_arr.col_header = 'Documents: '
    d_arr.subcol_headers = ['Document', 'Cosine']
    d_arr.col_len = print_len

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
    k_arr = row_cosines(top, mat, norms=norms)
    k_arr = enum_sort(k_arr, filter_nan=filter_nan)

    # Label data
    k_arr = k_arr.view(LabeledColumn)
    k_arr.col_header = 'Topics: ' + ', '.join([str(t) for t in topics])
    k_arr.subcol_headers = ['Topic', 'Cosine']
    k_arr.col_len = print_len

    return k_arr



def simmat_words(corp, matrix, word_list, norms=None):
    """
    """
    indices, words = zip(*[res_word_type(corp, word) 
                           for word in word_list])
    indices, words = np.array(indices), np.array(words)

    sm = row_cos_mat(indices, matrix, norms=norms, fill_tril=True)
    sm = sm.view(IndexedSymmArray)
    sm.labels = words
    
    return sm



def simmat_documents(corp, matrix, context_type, doc_list, norms=None):
    """
    """
    label_name = doc_label_name(context_type)

    indices, labels = zip(*[res_doc_type(corp, context_type, label_name, doc) 
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
