import numpy as np
from scipy.sparse import issparse

from vsm import (enum_sort, row_normalize, map_strarr, 
                 isfloat, isint, isstr)
from vsm.util.corpustools import word_tokenize
from vsm import model

from similarity import row_cosines, row_cos_mat

from labeleddata import *


def def_label_fn(metadata):
    """
    """
    names = [name for name in metadata.dtype.names if name.endswith('_label')]
    labels = [', '.join([x[n] for n in names]) for x in metadata]
    
    return np.array(labels)


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
    if issparse(mat):
        rows = mat.tocsr()[words].toarray()
    else:
        rows = mat[words]
    word = np.average(rows, weights=weights, axis=0)[np.newaxis, :]
    
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
    d_arr = row_cosines(doc, mat, norms=norms, filter_nan=filter_nan)

    # Label data
    if as_strings:
        md = corp.view_metadata(tok_name)
        docs = label_fn(md)
        d_arr = map_strarr(d_arr, docs, k='i', new_k='doc')
        
    d_arr = d_arr.view(LabeledColumn)
    d_arr.col_header = 'Topics: ' + ', '.join([str(t) for t in topics])
    d_arr.subcol_headers = ['Document', 'Prob']
    d_arr.col_len = print_len

    return d_arr


def sim_doc_doc(corp, mat, tok_name, doc_or_docs, weights=None,
                norms=None, filter_nan=True, print_len=10, 
                label_fn=def_label_fn, as_strings=True):
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
    if issparse(mat):
        rows = mat.tocsr()[docs].toarray()
    else:
        rows = mat[docs]
    doc = np.average(rows, weights=weights, axis=0)[np.newaxis, :]

    # Compute cosines
    d_arr = row_cosines(doc, mat, norms=norms, filter_nan=filter_nan)

    # Label data
    if as_strings:
        md = corp.view_metadata(tok_name)
        docs = label_fn(md)
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



def doc_label_name(tok_name):
    """
    """
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
