import numpy as np

from vsm import isstr, isint



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
                    
            

def res_word_type(corp, word):
    """
    If `word` is a string, performs a look up for its associated
    integer. Otherwise, stringifies `word`. 

    Returns an integer, string pair: (<word index>, <word label>).
    """
    if isstr(word):
        return corp.words_int[word], word

    return word, str(word)


def res_top_type(topic_or_topics):
    """
    If `topic_or_topics` is an int, then returns it in a list.
    """
    if isint(topic_or_topics):
        topic_or_topics = [topic_or_topics]

    return topic_or_topics
