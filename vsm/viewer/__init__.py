import numpy as np

from vsm import isstr



def def_label_fn(metadata):
    """
    """
    names = [name for name in metadata.dtype.names if name.endswith('_label')]
    labels = [', '.join([x[n] for n in names]) for x in metadata]
    
    return np.array(labels)



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
    if isstr(doc):
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
                    
            

def res_word_type(corp, word):
    """
    If `word` is a string, performs a look up for its associated
    integer. Otherwise, stringifies `word`. 

    Returns an integer, string pair: (<word index>, <word label>).
    """
    if isstr(word):
        return corp.words_int[word], word

    return word, str(word)
