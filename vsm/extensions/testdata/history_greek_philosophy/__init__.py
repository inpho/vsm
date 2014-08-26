import os, json


__all__ = [ 'doc_files', 'doc_meta_file',
            'documents', 'document_metadata',
            'corpus', 'paragraphs', 'doc_label_fn' ]



_doc_files = [ 'frontmatter.json', 'chapter1.json', 'chapter2.json',
               'chapter3.json', 'chapter4.json', 'chapter5.json',
               'chapter6.json', 'chapter7.json', 'chapter8.json',
               'chapter9.json', 'chapter10.json', 'chapter11.json',
               'chapter12.json', 'chapter13.json', 'chapter14.json',
               'chapter15.json', 'chapter16.json', 'chapter17.json',
               'chapter18.json', 'chapter19.json', 'chapter20.json',
               'chapter21.json', 'chapter22.json', 'backmatter.json' ]

doc_files = [os.path.join(os.path.dirname(__file__), f) 
                  for f in _doc_files]


doc_meta_file = os.path.join(os.path.dirname(__file__), 'doc_meta.json')


def document_metadata():
    """Returns an iterator over document metadata in corpus.

    """
    with open(doc_meta_file, 'r') as f:
        doc_meta_all = json.load(f)
    for docs_meta in doc_meta_all:
        for doc_meta in docs_meta:
            yield doc_meta


def documents():
    """Returns an iterator over documents paired with their metadata.

    """
    m = document_metadata()

    for doc_file in doc_files:
        with open(doc_file, 'r') as f:
            docs = json.load(f)
        for doc in docs:
            yield doc, m.next()


def paragraphs():
    """Returns iterator over paragraphs and associated metadata.

    """
    import copy
    import vsm.ext.corpusbuilders.util as util

    docs = documents()
    for doc, meta in docs:
        p = 0
        pars = util.paragraph_tokenize(doc)
        for par in pars:
            par_meta = copy.deepcopy(meta)
            par_meta['paragraph'] = p
            p += 1
            yield par, par_meta
 

def corpus(doc_type='document', unidecode=True, nltk_stop=True, 
           stop_freq=0, add_stop=None):
    """Returns Corpus object containing text data and metadata.

    """
    from vsm.ext.corpusbuilders import corpus_from_strings

    if doc_type=='document':
        docs = documents()
    elif doc_type=='paragraphs':
        docs = paragraphs()
    else:
        raise Exception('Unrecognized document type given.')

    docs, meta = zip(*list(docs))

    return corpus_from_strings(docs, meta, 
                               unidecode=unidecode, 
                               nltk_stop=nltk_stop, 
                               stop_freq=stop_freq, 
                               add_stop=add_stop)


def doc_label_fn(metadata):
    label = metadata['part_of_book']
    return label
