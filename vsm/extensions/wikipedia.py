#
# Takes output from
# https://github.com/jodaiber/Annotated-WikiExtractor and builds a
# Corpus object
#

import os
import json
import re
import errno
import shutil

import numpy as np
import unidecode
import nltk

from vsm.corpus import Corpus
from vsm.extensions.corpusbuilders.util import word_tokenize, apply_stoplist


__all__ = [ 'corpus_from_wikipedia', 'url_label_fn', 'title_label_fn' ]



def make_path(path):
    """
    Emulates mkdir -p
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if not (e.errno == errno.EEXIST and os.path.isdir(path)):
            raise


def build_tree(build_dir='build'):
    """
    """
    make_path(build_dir)
    build_dir = os.path.abspath(build_dir)

    corpus_dir = os.path.join(build_dir, 'corpus')
    make_path(corpus_dir)

    metadata_dir = os.path.join(build_dir, 'metadata')
    make_path(metadata_dir)

    stats_dir = os.path.join(build_dir, 'stats')
    make_path(stats_dir)


def src_filelist(src_dir):
    """
    """
    src_dir = os.path.abspath(src_dir)
    src_subdirs = os.listdir(src_dir)
    src_subdirs = [os.path.join(src_dir, d) for d in src_subdirs]
    src_files = []
    for d in src_subdirs:
        files = os.listdir(d)
        for f in files:
            src_files.append(os.path.join(d, f))

    return src_files


def corpus_filelist(build_dir='build'):
    """
    """
    corpus_dir = os.path.abspath(os.path.join(build_dir, 'corpus'))
    build_files = os.listdir(corpus_dir)
    return [os.path.join(corpus_dir, f) for f in build_files]


def metadata_filelist(build_dir='build'):
    """
    """
    metadata_dir = os.path.abspath(os.path.join(build_dir, 'metadata'))
    build_files = os.listdir(metadata_dir)
    return [os.path.join(metadata_dir, f) for f in build_files]



def init_corpus_dir(build_dir='build', src_dir=None, filenames=None):
    """
    """
    if filenames==None:
        filenames = src_filelist(src_dir)
    
    for filename in filenames:
        filename_split = os.path.split(filename)
        src_subdir = os.path.split(filename_split[0])[1] 
        file_out = os.path.join(build_dir, 'corpus', 
                                src_subdir + filename_split[1] + '.text.json')

        texts = []
        with open(filename, 'r') as f:
            records = f.read()
            records = records.split('\n')
        for r in records:
            if len(r) > 0:
                texts.append(json.loads(r)['text'])

        with open(file_out, 'w') as f:
            json.dump(texts, f)


    
def init_metadata_dir(build_dir='build', src_dir=None, filenames=None):
    """
    """
    if filenames==None:
        filenames = src_filelist(src_dir)
    
    for filename in filenames:
        filename_split = os.path.split(filename)
        src_subdir = os.path.split(filename_split[0])[1] 
        file_out = os.path.join(build_dir, 'metadata', 
                                src_subdir + filename_split[1] + '.metadata.json')

        metadata = []
        with open(filename, 'r') as f:
            records = f.read()
            records = records.split('\n')
        for r in records:
            if len(r) > 0:
                metadata.append(json.loads(r)['url'])

        with open(file_out, 'w') as f:
            json.dump(metadata, f)


def trim_urls(build_dir='build', filenames=None):
    """
    """
    if filenames==None:
        filenames = metadata_filelist(build_dir)
    
    for filename in filenames:
        with open(filename, 'r+') as f:
            urls = json.load(f)
            for i in xrange(len(urls)):
                urls[i] = os.path.split(urls[i])[1]
            f.seek(0)
            json.dump(urls, f)
            f.truncate()


def tokenize(build_dir='build', filenames=None):
    """
    """
    if filenames==None:
        filenames = corpus_filelist(build_dir)
    
    for filename in filenames:
        with open(filename, 'r+') as f:
            texts = json.load(f)
            texts = map(word_tokenize, texts)
            f.seek(0)
            json.dump(texts, f)
            f.truncate()


def stoplist(build_dir='build', filenames=None, nltk_stop=True, add_stop=None, 
             stop_freq=2, word_counts=None):
    """
    """
    stopwords = set()
    if nltk_stop:
        for w in nltk.corpus.stopwords.words('english'):
            stopwords.add(w)
    if add_stop:
        for w in add_stop:
            stopwords.add(w)

    if stop_freq > 0:
        for w,c in word_counts.iteritems():
            if c <= stop_freq:
                stopwords.add(w)

    def stop(text, stopwords=stopwords):
        return [w for w in text if w not in stopwords]

    if filenames==None:
        filenames = corpus_filelist(build_dir)
    
    for filename in filenames:
        with open(filename, 'r+') as f:
            texts = json.load(f)
            texts = map(stop, texts)
            f.seek(0)
            json.dump(texts, f)
            f.truncate()


def vocabulary(build_dir='build', filenames=None):
    """
    """
    if filenames==None:
        filenames = corpus_filelist(build_dir)

    vocab = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            texts = json.load(f)
        vocab.update(*map(set, texts))

    vocab = list(vocab)
    
    file_out = os.path.join(build_dir, 'vocabulary.json')
    with open(file_out, 'w') as f:
        json.dump(vocab, f)

    return vocab


def word_counts(build_dir='build', filenames=None):
    """
    """
    if filenames==None:
        filenames = corpus_filelist(build_dir)

    def count(text):
        wc = {}
        for w in text:
            if w in wc:
                wc[w] += 1
            else:
                wc[w] = 1
        return wc

    def update_wc(wc, new_wc):
        for w in new_wc:
            if w in wc:
                wc[w] += new_wc[w]
            else:
                wc[w] = new_wc[w]
        return wc

    word_counts = {}
    for filename in filenames:
        with open(filename, 'r') as f:
            texts = json.load(f)
        reduce(update_wc, map(count, texts), word_counts)

    file_out = os.path.join(build_dir, 'stats', 'word_counts.json')
    with open(file_out, 'w') as f:
        json.dump(word_counts, f)

    return word_counts


def global_indices(build_dir='build', filenames=None):
    """
    """
    if filenames==None:
        filenames = corpus_filelist(build_dir)

    lengths = []
    for filename in filenames:
        with open(filename, 'r') as f:
            texts = json.load(f)
        lengths += [len(t) for t in texts]

    indices = np.cumsum(lengths)

    file_out = os.path.join(build_dir, 'indices.npy')
    with open(file_out, 'w') as f:
        np.save(f, indices)

    return indices


def encode_corpus(words, build_dir='build', filenames=None):
    """
    """
    words_int = dict((words[i], i) for i in xrange(len(words)))

    file_out = os.path.join(build_dir, 'words_int.json')
    with open(file_out, 'w') as f:
        json.dump(words_int, f)

    def map_fn(text):
        return [words_int[w] for w in text]

    if filenames==None:
        filenames = corpus_filelist(build_dir)
    
    for filename in filenames:
        with open(filename, 'r+') as f:
            texts = json.load(f)
            texts = map(map_fn, texts)
            f.seek(0)
            json.dump(texts, f)
            f.truncate()
    
    return words_int


def reduce_corpus(corpus_len, build_dir='build', filenames=None):
    """
    """
    corpus = np.empty((corpus_len,), dtype='i')
    
    if filenames==None:
        filenames = corpus_filelist(build_dir)
    
    idx = 0
    for filename in filenames:
        with open(filename, 'r') as f:
            texts = json.load(f)
            for text in texts:
                next_idx = idx+len(text)
                corpus[idx:next_idx] = text
                idx = next_idx

    file_out = os.path.join(build_dir, 'corpus.npy')
    with open(file_out, 'w') as f:
        np.save(f, corpus)

    return corpus


def reduce_metadata(indices, build_dir='build', filenames=None):
    """
    """
    if filenames==None:
        filenames = metadata_filelist(build_dir)
    
    urls = []
    for filename in filenames:
        with open(filename, 'r') as f:
            urls += json.load(f)

    urls = [ unidecode.unidecode(url) for url in urls ]
    
    url_dtype = np.array(urls).dtype
    dtype = [('idx', 'i'), ('url', url_dtype)]
    metadata = np.array(zip(indices, urls), dtype=dtype)

    file_out = os.path.join(build_dir, 'metadata.npy')
    with open(file_out, 'w') as f:
        np.save(f, metadata)

    return metadata


def make_corpus_obj(build_dir='build'):
    """
    """
    with open(os.path.join(build_dir, 'vocabulary.json'), 'r') as f:
        words = json.load(f)
    with open(os.path.join(build_dir, 'words_int.json'), 'r') as f:
        words_int = json.load(f)
    corpus = np.load(os.path.join(build_dir, 'corpus.npy'))
    metadata = np.load(os.path.join(build_dir, 'metadata.npy'))
        
    corp_obj = Corpus([])
    corp_obj.words = np.array(words)
    corp_obj.words_int = words_int
    corp_obj.corpus = corpus
    corp_obj.context_data = [ metadata ]
    corp_obj.context_types = [ 'document' ]
    
    file_out = os.path.join(build_dir, 'corpus-object.npz')
    corp_obj.save(file_out)
    
    return corp_obj


def clean(build_dir='build'):
    """
    """
    shutil.rmtree(build_dir)


def corpus_from_wikipedia(src_dir, build_dir='build',
                          nltk_stop=True, stop_freq=2, add_stop=None,
                          trim_urls=True):
    """
    """
    src_files = src_filelist(src_dir)

    build_tree(build_dir=build_dir)

    init_corpus_dir(build_dir=build_dir, filenames=src_files)
    init_metadata_dir(build_dir=build_dir, filenames=src_files)

    corpus_files = corpus_filelist(build_dir=build_dir)
    metadata_files = metadata_filelist(build_dir=build_dir)
    tokenize(filenames=corpus_files)

    wc = word_counts(filenames=corpus_files)

    stoplist(filenames=corpus_files, nltk_stop=nltk_stop, 
             add_stop=add_stop, stop_freq=stop_freq, word_counts=wc)
    
    indices = global_indices(filenames=corpus_files)
    corpus_len = indices[-1]

    words = vocabulary(filenames=corpus_files)
    words_int = encode_corpus(words, filenames=corpus_files)

    if trim_urls:
        trim_urls(build_dir=build_dir)

    reduce_corpus(corpus_len, filenames=corpus_files)
    reduce_metadata(indices, filenames=metadata_files)

    corp_obj = make_corpus_obj(build_dir=build_dir)
    
    return corp_obj


def url_label_fn(metadata):
    return metadata['url']


def title_label_fn(metadata):
    return os.path.split(metadata['url'])[1]
