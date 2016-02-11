import numpy as np

from scipy.sparse import issparse, csr_matrix, coo_matrix

from vsm.spatial import angle, JS_dist
from vsm.structarr import *
from types import *
from labeleddata import *


__all__ = ['def_label_fn', 'doc_label_name',
           'dismat_doc', 'dismat_top', 'dismat_word', 'dist_doc_doc',
           'dist_word_doc', 'dist_word_top', 'dist_word_word', 
           'dist_top_doc', 'dist_top_top']

# 
# A default function for constructing document labels from corpus
# metadata
#


def def_label_fn(metadata):
    """
    Takes metadata and returns an array of strings that are 
    strings constructed with all "labels" in the metdata.

    :param metadata: Metadata most likely retrieved from
        Corpus.view_metadata(ctx_type).
    :type metadata: array

    :returns: Array of strings. Each entry is a string with different
        metadata separated by ', '. For example, '<page_label>, <book_label>'
        can be an entry of the array.

    :See Also: :meth:`vsm.corpus.Corpus.view_metadata`

    **Examples**

    >>> c = random_corpus(5, 5, 1, 3, context_type='ctx', metadata=True) 
    >>> meta = c.view_metadata('ctx')
    >>> meta
    array([(2, 'ctx_0'), (3, 'ctx_1'), (4, 'ctx_2'), (5, 'ctx_3')], 
          dtype=[('idx', '<i8'), ('ctx_label', '|S5')])
    >>> def_label_fn(meta)
    array(['ctx_0', 'ctx_1', 'ctx_2', 'ctx_3'], 
          dtype='|S5')
    """
    names = [name for name in metadata.dtype.names if name.endswith('_label')]
    labels = [', '.join([x[n] for n in names]) for x in metadata]
    
    return np.array(labels)


def doc_label_name(context_type):
    """
    Takes a string `context_type` and makes that a standard 'label' string.
    This functions is useful for figuring out the specific label name in 
    metadata given the context_type.

    :param context_type: A type of tokenization.
    :type context_type: string

    :returns: label name for `context_type` as string.
    """
    return context_type + '_label'


# 
# wrappers of distance functions
# 


def dist_word_word(word_or_words, corp, mat, weights=[],
                   as_strings=True, print_len=20,
                   filter_nan=True, dist_fn=angle, order='i'):
    """
    Computes and sorts the distances between a word or list of words
    and every word. If weights are provided, the word list is
    represented as the weighted average of the words in the list. If
    weights are not provided, the arithmetic mean is used.

    The columns of `mat` are assumed to be vector representations of
    words.

    """
    # Resolve `word_or_words`
    if isstr(word_or_words):
        word_or_words = [word_or_words]
    words, labels = zip(*[res_word_type(corp, w) for w in word_or_words])
    words, labels = list(words), list(labels)

    if len(words)==0:
        raise Exception('At least one target word is needed.')

    # Generate pseudo-word
    if len(weights) == 0:
        weights=None
    if issparse(mat):
        cols = mat.tocsc()[:,words].toarray()
        word = np.average(cols, weights=weights, axis=1)[np.newaxis,:]
        word = csr_matrix(word)
    else:
        cols = mat[:,words]
        word = np.average(cols, weights=weights, axis=1)[np.newaxis,:]
    
    # Compute distances
    w_arr = dist_fn(word, mat)

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
    w_arr.subcol_headers = ['Word', 'Distance']
    w_arr.col_len = print_len

    return w_arr


def dist_doc_doc(doc_or_docs, corp, context_type, mat, weights=[],
                 filter_nan=True, print_len=10,
                 label_fn=def_label_fn, as_strings=True,
                 dist_fn=angle, order='i'):
    """
    Computes and sorts the distances between a document or list of
    documents and every document. If weights are provided, the document
    list is represented as the weighted average of the documents in the list.
    If weights are not provided, the arithmetic mean is used.

    The columns of `mat` are assumed to represent documents.
    """
    # Resolve `doc_or_docs`
    label_name = doc_label_name(context_type)    
    if (isstr(doc_or_docs) or isint(doc_or_docs) 
        or isinstance(doc_or_docs, dict)):
        docs = [res_doc_type(corp, context_type, label_name, doc_or_docs)[0]]
    else:
        docs = [res_doc_type(corp, context_type, label_name, d)[0] 
                for d in doc_or_docs]

    if len(docs)==0:
        raise Exception('At least one target document is needed.')

    # Generate pseudo-document
    if len(weights) == 0:
        weights=None
    if issparse(mat):
        cols = mat.tocsc()[:,docs].toarray()
        doc = np.average(cols, weights=weights, axis=1)[np.newaxis,:]
        doc = csr_matrix(doc)
    else:
        cols = mat[:,docs]
        doc = np.average(cols, weights=weights, axis=1)

    # Compute distances
    d_arr = dist_fn(doc, mat)

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
    d_arr.subcol_headers = ['Document', 'Distance']
    d_arr.col_len = print_len

    return d_arr


def dist_word_doc(word_or_words, corp, context_type, mat, weights=[], 
                  filter_nan=True, print_len=10,
                  label_fn=def_label_fn, as_strings=True,
                  dist_fn=angle, order='i'):
    """
    Computes distances between a word or a list of words to every
    document and sorts the results. The function constructs a
    pseudo-document vector from `word_or_words` and `weights`: the
    vector representation is non-zero only if the corresponding word
    appears in the list. If `weights` are not given, `1` is assigned
    to each word in `word_or_words`.
    """
    # Resolve `word_or_words`
    if isstr(word_or_words):
        word_or_words = [word_or_words]
    words, labels = zip(*[res_word_type(corp, w) for w in word_or_words])
    words, labels = list(words), list(labels)

    if len(words)==0:
        raise Exception('At least one target word is needed.')

    # Generate pseudo-document
    doc = np.zeros(mat.shape[0], dtype=np.float)
    if len(weights) == 0:
        doc[words] = np.ones(len(words))
    else:
        doc[words] = weights
    if issparse(mat):
        doc = coo_matrix(doc)

    # Compute distances
    d_arr = dist_fn(doc.T, mat)

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
    d_arr.col_header = 'Words: '
    d_arr.subcol_headers = ['Document', 'Distance']
    d_arr.col_len = print_len

    return d_arr


def dist_word_top(word_or_words, corp, mat, weights=[], 
                  print_len=10, filter_nan=True, 
                  dist_fn=JS_dist, order='i'):
    """
    Computes distances between a word or a list of words to every topic
    and sorts the results. The function treats the query words as a pseudo-topic
    that assign to those words non-zero probability masses specified by `weight`.
    Otherwise equal probability is assigned to each word in `word_or_words`. 
    """
    # Resolve `word_or_words`
    if isstr(word_or_words):
        word_or_words = [word_or_words]
    words, labels = zip(*[res_word_type(corp, w) for w in word_or_words])
    words, labels = list(words), list(labels)

    if len(words)==0:
        raise Exception('At least one target word is needed.')

    # Generate pseudo-topic
    top = np.zeros(mat.shape[0], dtype=np.float)
    if len(weights) == 0:
        top[words] = np.ones(len(words))
    else:
        top[words] = weights

    # Compute distances
    k_arr = dist_fn(top, mat)
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
    k_arr.subcol_headers = ['Topic', 'Distance']
    k_arr.col_len = print_len

    return k_arr


def dist_top_doc(topic_or_topics, mat, corp, context_type, weights=[],
                 print_len=10, filter_nan=True, 
                 label_fn=def_label_fn, as_strings=True,
                 dist_fn=JS_dist, order='i'):
    """
    Takes a topic or list of topics (by integer index) and returns a
    list of documents sorted by distance.
    
    The columns of `mat` are assumed to represent documents.    
    """
    topics = res_top_type(topic_or_topics)

    if len(topics)==0:
        raise Exception('At least one target topic is needed.')

    # Generate pseudo-document
    doc = np.zeros(mat.shape[0], dtype=np.float)
    if len(weights) == 0:
        doc[topics] = np.ones(len(topics))
    else:
        doc[topics] = weights
    doc /= doc.sum()

    # Compute similarites/divergences
    d_arr = dist_fn(doc, mat)

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
    d_arr.subcol_headers = ['Document', 'Distance']
    d_arr.col_len = print_len

    return d_arr


def dist_top_top(mat, topic_or_topics, weights=[], 
                 print_len=10, filter_nan=True, 
                 dist_fn=JS_dist, order='i'):
    """
    Takes a topic or list of topics (by integer index) and returns
    a list of topics sorted by the distances between a given topic
    and every topic.

    The columns of `mat` are assumed to be probability distributions
    (namely, topics).
    """
    topics = res_top_type(topic_or_topics)

    if len(topics)==0:
        raise Exception('At least one target topic is needed.')

    # Generate pseudo-topic
    if len(weights) == 0:
        weights = None
    top = np.average(mat[:,topics], weights=weights, axis=1)

    # Compute distances
    k_arr = dist_fn(top, mat)
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
    k_arr.subcol_headers = ['Topic', 'Distance']
    k_arr.col_len = print_len

    return k_arr


def dismat_word(word_list, corp, mat, dist_fn=angle):
    """
    Calculates a distance matrix for a given list of words.

    The columns of `mat` are assumed to represent words.
    """
    indices, words = zip(*[res_word_type(corp, word) 
                           for word in word_list])
    indices, words = np.array(indices), np.array(words)

    mat = mat[:,indices]

    dm = dist_fn(mat.T, mat)
    if np.isscalar(dm):
        dm = np.array([dm])
    dm = dm.view(IndexedSymmArray)
    dm.labels = words
    
    return dm


#TODO: Abstract the label creation correctly
def dismat_doc(doc_list, corp, context_type, mat, dist_fn=angle):
    """
    Calculates a distance matrix for a given list of documents.
    
    The columns of `mat` are assumed to be probability distributions
    (namely, over topics).
    """
    label_name = doc_label_name(context_type)

    indices, labels = zip(*[res_doc_type(corp, context_type, label_name, doc) 
                            for doc in doc_list])
    indices, labels = np.array(indices), np.array(labels)

    mat = mat[:,indices]

    dm = dist_fn(mat.T, mat)
    if np.isscalar(dm):
        dm = np.array([dm])
    dm = dm.view(IndexedSymmArray)
    dm.labels = labels
    
    return dm


def dismat_top(topics, mat, dist_fn=JS_dist):
    """
    Calculates a distance matrix for a given list of topics.
    
    The columns of `mat` are assumed to be probability distributions
    (namely, topics).
    """
    mat = mat[:,topics]

    dm = dist_fn(mat.T, mat)
    if np.isscalar(dm):
        dm = np.array([dm])
    dm = dm.view(IndexedSymmArray)
    dm.labels = [str(k) for k in topics]
    
    return dm
