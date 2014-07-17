import os, json

from vsm.corpus.mapcorpus import(
    autogen_corpus as _autogen_corpus,
    count_docs as _count_docs,
    document as _document,
    doc_metadata as _doc_metadata,
    corp_metadata as _corp_metadata,
    update_doc_metadata as _update_doc_metadata
    )

__all__ = [ 'history_greek_philosophy', 'doc_label_fn', 
            'documents_by_label' ]


_path = os.path.join(os.path.dirname(__file__), 
                     'short-history-greek-philosophy.json')

_archive = _autogen_corpus([_path], archive=True)

for i in xrange(1, _count_docs(archive=_archive)-1):
    _archive = _update_doc_metadata({ 'part_of_book': 'Chapter %d' % i},
                                    i, archive=_archive)

_archive = _update_doc_metadata({ 'part_of_book': 'Front Matter' }, 
                                0, archive=_archive)
_archive = _update_doc_metadata({ 'part_of_book': 'Back Matter' }, 
                                _count_docs(archive=_archive)-1, 
                                archive=_archive)

history_greek_philosophy = _archive

def doc_label_fn(global_idx, metadata):
    label = metadata['part_of_book']
    return label

def documents_by_label():
    return [doc_label_fn(i, _doc_metadata(i, archive=_archive)) 
            for i in xrange(_count_docs(archive=_archive))]


# def document(global_idx):
#     return _document(global_idx, archive=_archive)

# def doc_metadata(global_idx):
#     return _doc_metadata(global_idx, archive=_archive)

# def corp_metadata(global_idx):
#     return _corp_metadata(archive=_archive)

# def count_docs():
#     return _count_docs(archive=_archive) 

# def document_label(global_idx):
#     return _doc_label_fn(global_idx, doc_metadata(global_idx))


