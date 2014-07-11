"""
"""

import os
import errno
import copy
import json
import tempfile
import shutil




#
# Functions to generate corpus file tree and archive stubs 
#

def make_path(path):
    """
    Emulates mkdir -p
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if not (e.errno == errno.EEXIST and os.path.isdir(path)):
            raise


def init_corpus_dir(corpus_dir):
    """
    """
    make_path(corpus_dir)
    make_path(os.path.join(corpus_dir, 'documents'))
    make_path(os.path.join(corpus_dir, 'metadata'))


def init_doc_meta(src=None):
    """
    """
    if src:
        doc_meta = copy.deepcopy(src)
        doc_meta['file_idx'] = None
        doc_meta['local_idx'] = None
        doc_meta['global_idx'] = None
    else:
        doc_meta = { 'user_defined': {}, 
                     'file_idx': None,
                     'local_idx': None,
                     'global_idx': None }

    return doc_meta


def init_corp_meta(src=None):
    """Generates a minimal corpus metadata data structure. If `src` is
    provided, all metadata from `src` not under the key `files` will
    be copied to the generated dictionary.

    """
    if src:
        corp_meta = copy.deepcopy(src)
        corp_meta['files'] = []
    else:
        corp_meta = { 'user_defined': {}, 'files': [], 'doc_paths': [] }

    return corp_meta


def init_archive():
    """
    """
    return { 'documents': {}, 
             'metadata': { 'corpus.metadata.json': init_corp_meta() } }


# 
# Functions to retrieve and to write documents and metadata 
# 


def get_corp_meta(corpus_dir=None, archive=None):
    """
    """
    if corpus_dir:
        corp_meta_file = os.path.join(corpus_dir, 'metadata', 
                                      'corpus.metadata.json')
        with open(corp_meta_file, 'r') as f:
            return json.load(f)
    elif archive:
        return archive['metadata']['corpus.metadata.json']
    else:
        return


def get_docs(doc_file, corpus_dir=None, archive=None):
    """
    """
    if corpus_dir:
        doc_file = os.path.join(corpus_dir, 'documents', doc_file)
        with open(doc_file, 'r') as f:
            return json.load(f)
    elif archive:
        return archive['documents'][doc_file]
    else:
        return

    
def get_doc_meta(meta_file, corpus_dir=None, archive=None):
    """
    """
    if corpus_dir:
        meta_file = os.path.join(corpus_dir, 'metadata', meta_file)
        with open(meta_file, 'r') as f:
            return json.load(f)
    elif archive:
        return archive['metadata'][meta_file]
    else:
        return


def write_docs(docs, doc_file, corpus_dir=None, archive=None):
    """
    """
    if corpus_dir:
        doc_file = os.path.join(corpus_dir, 'documents', doc_file)
        with open(doc_file, 'w') as f:
            json.dump(docs, f)
            f.truncate()
        return
    elif archive:
        archive['documents'][doc_file] = docs
        return archive
        

def write_doc_meta(docs_meta, doc_meta_file, corpus_dir=None, archive=None):
    """
    """
    if corpus_dir:
        doc_meta_file = os.path.join(corpus_dir, 'metadata', doc_meta_file)
        with open(doc_meta_file, 'w') as f:
            json.dump(docs_meta, f, sort_keys=True)
            f.truncate()
        return
    elif archive:
        archive['metadata'][doc_meta_file] = docs_meta
        return archive


def write_corp_meta(corp_meta, corpus_dir=None, archive=None):
    """
    """
    if corpus_dir:
        corp_meta_file = os.path.join(corpus_dir, 'metadata', 
                                      'corpus.metadata.json')
        with open(corp_meta_file, 'w') as f:
            json.dump(corp_meta, f, sort_keys=True)
            f.truncate()
        return
    elif archive:
        archive['metadata']['corpus.metadata.json'] = corp_meta
        return archive


# 
# Corpus navigation
# 

def count_docs(corpus_dir=None, archive=None):
    
    corp_meta = get_corp_meta(corpus_dir=corpus_dir, archive=archive)
    return len(corp_meta['doc_paths'])


def document(global_idx, corpus_dir=None, archive=None):

    corp_meta = get_corp_meta(corpus_dir=corpus_dir, archive=archive)
    file_idx, local_idx = corp_meta['doc_paths'][global_idx]
    doc_file, meta_file = corp_meta['files'][file_idx]
    docs = get_docs(doc_file, corpus_dir=corpus_dir, archive=archive)
    
    return docs[local_idx]


def documents(indices, corpus_dir=None, archive=None):
    """
    `indices` is a list of (global) document indices
    """
    corp_meta = get_corp_meta(corpus_dir=corpus_dir, archive=archive)
    out = []
    for global_idx in indices:
        file_idx, local_idx = corp_meta['doc_paths'][global_idx]
        doc_file, meta_file = corp_meta['files'][file_idx]
        docs = get_docs(doc_file, corpus_dir=corpus_dir, archive=archive)
        out.append(docs[local_idx]) 
    return out


def doc_metadata(global_idx, corpus_dir=None, archive=None, user_defined=True):

    corp_meta = get_corp_meta(corpus_dir=corpus_dir, archive=archive)
    file_idx, local_idx = corp_meta['doc_paths'][global_idx]
    doc_file, meta_file = corp_meta['files'][file_idx]
    docs_meta = get_doc_meta(meta_file, corpus_dir=corpus_dir, archive=archive)
    
    if user_defined:
        return docs_meta[local_idx]['user_defined']
    return docs_meta[local_idx]


def corp_metadata(corpus_dir=None, archive=None, user_defined=True):
    
    corp_meta = get_corp_meta(corpus_dir=corpus_dir, archive=archive)

    if user_defined:
        return corp_meta[local_idx]['user_defined']
    return corp_meta[local_idx]


#
# Tools to add user-defined metadata
# 

def update_corp_metadata(metadata, corpus_dir=None, archive=None, 
                         remove_keys=False):
    """
    `metadata` is a dictionary.
    """
    corp_meta = get_corp_meta(corpus_dir=corpus_dir, archive=archive)
    if remove_keys:
        corp_meta['user_defined'] = metadata
    else:
        for key in metadata:
            corp_meta['user_defined'][key] = metadata[key]
            
    return write_corp_meta(corp_meta, corpus_dir=corpus_dir,
                           archive=archive)


def update_doc_metadata(metadata, global_idx, 
                        corpus_dir=None, archive=None, remove_keys=False):
    """
    `metadata` is a dictionary.
    """
    corp_meta = get_corp_meta(corpus_dir=corpus_dir, archive=archive)
    file_idx, local_idx = corp_meta['doc_paths'][global_idx]
    doc_file, meta_file = corp_meta['files'][file_idx]

    docs_meta = get_doc_meta(meta_file, corpus_dir=corpus_dir, 
                             archive=archive)
    if remove_keys:
        docs_meta[local_idx]['user_defined'] = metadata 
    else:
        for key in metadata:
            docs_meta[local_idx]['user_defined'][key] = metadata[key]

    return write_doc_meta(docs_meta, meta_file, 
                          corpus_dir=corpus_dir, archive=archive)

    

# 
# Conversion between corpus file trees and archives
# 

def file_tree_to_archive(corpus_dir):
    """
    """
    archive = init_archive()

    corp_meta = get_corp_meta(corpus_dir=corpus_dir)
    archive['metadata']['corpus.metadata.json'] = corp_meta

    for doc_file, meta_file in corp_meta['files']:
        docs = get_docs(doc_file, corpus_dir=corpus_dir)
        archive['documents'][doc_file] = docs

        docs_meta = get_doc_meta(meta_file, corpus_dir=corpus_dir)
        archive['metadata'][meta_file] = docs_meta

    return archive


def archive_to_file_tree(archive, corpus_dir):
    """
    """
    init_corpus_dir(corpus_dir)

    write_corp_meta(archive['metadata']['corpus.metadata.json'],
                    corpus_dir=corpus_dir)

    files = archive['metadata']['corpus.metadata.json']['files']
    for doc_file, meta_file in files:
        write_docs(archive['documents'][doc_file], doc_file, 
                   corpus_dir=corpus_dir)
        write_doc_meta(archive['metadata'][meta_file], meta_file, 
                       corpus_dir=corpus_dir)

    return


# 
# Global document indexing and bookkeeping
# 


def update_doc_paths(corpus_dir=None, archive=None):

    corp_meta = get_corp_meta(corpus_dir=corpus_dir, archive=archive)

    paths = {}
    for doc_file, meta_file in corp_meta['files']:
        docs_meta = get_doc_meta(meta_file, corpus_dir=corpus_dir, 
                                 archive=archive)
        for doc_meta in docs_meta:
            paths[doc_meta['global_idx']] = [ doc_meta['file_idx'], 
                                              doc_meta['local_idx'] ]

    corp_meta['doc_paths'] = [paths[i] for i in xrange(len(paths))]
            
    return write_corp_meta(corp_meta, corpus_dir=corpus_dir,
                           archive=archive)


def index_globally(corpus_dir=None, archive=None):
    """
    """
    corp_meta = get_corp_meta(corpus_dir=corpus_dir, archive=archive)

    global_idx = 0
    for doc_file, meta_file in corp_meta['files']:

        docs_meta = get_doc_meta(meta_file, corpus_dir=corpus_dir, 
                                 archive=archive)

        for local_idx in xrange(len(docs_meta)):
            docs_meta[local_idx]['global_idx'] = global_idx
            global_idx += 1

        write_doc_meta(docs_meta, meta_file, 
                       corpus_dir=corpus_dir, archive=archive)

    update_doc_paths(corpus_dir=corpus_dir, archive=archive)

    if corpus_dir:
        return
    return archive


#
# Autogeneration of a corpus tree or archive from document files of
# some sort
#

def derive_basename(src_file, suffix='', metadata=False):
    """
    """
    base_src = os.path.basename(src_file)
    ext = os.path.splitext
    if metadata:
        return ext(ext(base_src)[0])[0] + suffix + '.metadata.json'
    return ext(base_src)[0] + suffix + '.json'


def proc_doc_file(doc_file):
    """Opens a file named `doc_file` which stores documents either in
    JSON format or as text and returns the decoded contents.

    Allows the following JSON data structures: a string, an array of
    strings, an array of arrays of strings or an array of arrays of
    integers.

    Corresponding example use cases. The contents of `doc_file` is the
    text of a single document. The contents of `doc_file` is an
    ordered collection of document texts. The contents of `doc_file`
    is an ordered collection of documents which are stored as lists of
    words. The contents of `doc_file` is an ordered collection of
    documents which are stored as lists of integers.
    """
    with open(doc_file, 'r') as f:
        try:
            docs = json.load(f)
            if isinstance(docs, unicode):
                return [docs]
            elif isinstance(docs, list):  
                all_strings = reduce(lambda p, q: p and q,
                                     [isinstance(d, unicode) for d in docs])
                if all_strings:
                    return docs
                else:
                    all_lists = reduce(lambda p, q: p and q,
                                       [isinstance(d, list) for d in docs])
                    if all_lists:
                        all_strings = reduce(lambda p, q: p and q,
                                             [isinstance(w, unicode) 
                                              for d in docs for w in d])
                        all_integers = reduce(lambda p, q: p and q,
                                              [isinstance(w, int) 
                                               for d in docs for w in d])
                        if all_strings or all_integers:
                            return docs
                        else:
                            msg = '%s is JSON-decodable '\
                                  'but not as a JSON array of strings, '\
                                  'a JSON array of arrays of strings '\
                                  'or a JSON array of arrays of integers.' % doc_file
                            raise ValueError(msg)
            else:
                msg = '%s is JSON-decodable but not as a JSON array.' % doc_file
                raise ValueError(msg)
        
        except ValueError:
            f.seek(0)
            return [f.read()]


def autogen_corpus(doc_files, corpus_dir=None, archive=False):
    """
    """
    if corpus_dir:
        init_corpus_dir(corpus_dir)
    elif archive:
        archive = init_archive()
    else:
        return

    corp_meta = init_corp_meta()
    
    for file_idx, doc_src_file in enumerate(doc_files):

        doc_file = derive_basename(doc_src_file, suffix='')
        doc_meta_file = derive_basename(doc_src_file, suffix='', metadata=True)
        corp_meta['files'].append([doc_file, doc_meta_file])

        docs = proc_doc_file(doc_src_file)
        
        docs_meta = []
        for local_idx in xrange(len(docs)):
            doc_meta = init_doc_meta()
            doc_meta['file_idx'] = file_idx
            doc_meta['local_idx'] = local_idx
            docs_meta.append(doc_meta)

        write_docs(docs, doc_file, corpus_dir=corpus_dir, archive=archive)
        write_doc_meta(docs_meta, doc_meta_file, 
                       corpus_dir=corpus_dir, archive=archive)

    write_corp_meta(corp_meta, corpus_dir=corpus_dir, archive=archive)

    index_globally(corpus_dir=corpus_dir, archive=archive)

    if corpus_dir:
        return
    return archive


def autogen_corpus_from_string(string, corpus_dir=None, archive=False):
    """
    """
    try:
        f = tempfile.NamedTemporaryFile(delete=False)
        doc_file = f.name
        f.write(string)
        f.close()
        out = autogen_corpus([doc_file], corpus_dir=corpus_dir, archive=archive)
    finally:
        os.remove(doc_file)
    
    return out


# 
# Mapping over a corpus tree or an archive
# 

def map_corpus(fn, corpus_dir=None, archive=None, src_dir=None, 
               suffix='', reindex=True):
    """
    """
    if corpus_dir:
        if src_dir:
            init_corpus_dir(corpus_dir)
        else:
            src_dir = corpus_dir

    # Generate new corpus metadata
    corp_meta_src = get_corp_meta(corpus_dir=src_dir, archive=archive)
    corp_meta = init_corp_meta(src=corp_meta_src)
    for doc_src_file, meta_src_file in corp_meta_src['files']:
        doc_file = derive_basename(doc_src_file, suffix=suffix)
        doc_meta_file = derive_basename(meta_src_file, suffix=suffix, 
                                        metadata=True)
        corp_meta['files'].append([doc_file, doc_meta_file])
    write_corp_meta(corp_meta, corpus_dir=corpus_dir, archive=archive)

    # Map function over documents and their metadata
    for doc_src_file, meta_src_file in corp_meta_src['files']:

        docs_src = get_docs(doc_src_file, corpus_dir=src_dir, archive=archive)
        docs_meta_src = get_doc_meta(meta_src_file, corpus_dir=src_dir, 
                                     archive=archive)

        docs, docs_meta = fn(docs_src, docs_meta_src)

        doc_file = derive_basename(doc_src_file, suffix=suffix)
        write_docs(docs, doc_file, corpus_dir=corpus_dir, archive=archive)

        doc_meta_file = derive_basename(meta_src_file, suffix=suffix, 
                                        metadata=True)
        write_doc_meta(docs_meta, doc_meta_file, 
                       corpus_dir=corpus_dir, archive=archive)

    if reindex:
        index_globally(corpus_dir=corpus_dir, archive=archive)

    if corpus_dir:
        return
    return archive


# 
# Wrapper
# 


def corpus_map_fn(doc_fn, split=False):
    """
    """
    def fn(*args, **kwargs):
        docs_src = args[0]
        docs_meta_src = args[1]
        other_args = args[2:]
        
        docs = []
        docs_meta = []
        local_idx = 0
        for i in xrange(len(docs_src)):
            args_ = [docs_src[i]] + list(other_args)
            subdocs = doc_fn(*args_, **kwargs)
            if not split:
                subdocs = [subdocs]
            docs.extend(subdocs)

            for j in xrange(len(subdocs)):
                if split:
                    doc_meta = copy.deepcopy(docs_meta_src[i])
                    doc_meta['local_idx'] = local_idx
                    local_idx += 1
                    doc_meta['global_idx'] = None
                else:
                    doc_meta = docs_meta_src[i]
                docs_meta.append(doc_meta)

        return docs, docs_meta
        
    return fn

