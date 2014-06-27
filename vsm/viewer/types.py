import numpy as np


__all__ = ['isfloat', 'isint', 'isstr', 
           'res_doc_type', 'res_top_type', 'res_word_type']



# 
# Rudimentary type-checking fns
#


def isfloat(x):
    """
    Returns True if `x` is an instance of a float.
    """
    return (isinstance(x, np.inexact) or isinstance(x, np.float))


def isint(x):
    """
    Returns True if `x` is an instance of an int.
    """
    return (isinstance(x, np.integer) 
            or isinstance(x, int) or isinstance(x, long))


def isstr(x):
    """
    Returns True if `x` is an instance of a string.
    """
    return isinstance(x, basestring) or isinstance(x, np.flexible)


# 
# fns to resolve input polymorphism to the dist_*_* fns
#


def res_doc_type(corp, context_type, label_name, doc):
    """
    If `doc` is a string or a dict, performs a look up for its
    associated integer. If `doc` is a dict, looks for its label.
    Finally, if `doc` is an integer, stringifies `doc` for use as
    a label.
    
    Returns an integer, string pair: (<document index>, <document
    label>).
    """
    if isstr(doc):
        query = {label_name: doc}
        d = corp.meta_int(context_type, query)
    elif isinstance(doc, dict):
        d = corp.meta_int(context_type, doc)
        
        #TODO: Define an exception for failed queries in
        #vsm.corpus. Use it here.
        doc = corp.view_metadata(context_type)[label_name][d]
    else:
        d, doc = doc, str(doc)

    return d, doc
                    

def res_top_type(topic_or_topics):
    """
    If `topic_or_topics` is an int, then returns it in a list.
    """
    if isint(topic_or_topics):
        topic_or_topics = [topic_or_topics]

    return topic_or_topics


def res_word_type(corp, word):
    """
    If `word` is a string, performs a look up for its associated
    integer. Otherwise, stringifies `word`. 

    Returns an integer, string pair: (<word index>, <word label>).
    """
    if isstr(word):
        return corp.words_int[word], word

    return word, str(word)
