import os, json


__all__ = [ 'doc_files', 'doc_meta_file',
            'documents', 'document_metadata',
            'corpus', 'doc_label_fn' ]



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


def documents():
    """Returns an iterator over of documents in corpus.

    """
    for doc_file in doc_files:
        with open(doc_file, 'r') as f:
            docs = json.load(f)
        for doc in docs:
            yield doc


def document_metadata():
    """Returns an iterator over of document metadata in corpus.

    """
    with open(doc_meta_file, 'r') as f:
        doc_meta_all = json.load(f)
    for docs_meta in doc_meta_all:
        for doc_meta in docs_meta:
            yield doc_meta


def corpus(unidecode=True, nltk_stop=True, stop_freq=0, add_stop=None):
    """Returns Corpus object containing text data and metadata.

    """
    from vsm.ext.corpusbuilders import corpus_from_strings

    return corpus_from_strings(documents(), 
                               list(document_metadata()), 
                               unidecode=unidecode, 
                               nltk_stop=nltk_stop, 
                               stop_freq=stop_freq, 
                               add_stop=add_stop)


def doc_label_fn(global_idx, metadata):
    label = metadata['part_of_book']
    return label
