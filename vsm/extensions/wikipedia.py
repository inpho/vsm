#
# Takes output from
# https://github.com/jodaiber/Annotated-WikiExtractor and builds a
# Corpus object
#

import os
import json
import re
import unidecode as uni

import numpy as np

from vsm.corpus import Corpus
from vsm.extensions.corpusbuilders.util import word_tokenize, apply_stoplist



def article_url(filenames):
    
    for filename in filenames:
        with open(filename, 'r') as f:
            records = f.read()
        records = records.split('\n')
        for r in records:
            if len(r) > 0:
                d = json.loads(r)
                yield d['url']
    

def article_words(filenames):

    for filename in filenames:
        with open(filename, 'r') as f:
            records = f.read()
        records = records.split('\n')
        for r in records:
            if len(r) > 0:
                d = json.loads(r)
                words = word_tokenize(d['text'])
                yield words


def corpus_from_wikipedia(src_dir='.', unidecode=True,
                          nltk_stop=True, stop_freq=1, add_stop=None):
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
    
    indices = []
    corpus = []
    for a in article_words(src_files):
        corpus += a
        indices.append(len(a))
    indices = np.cumsum(indices)
    urls = list(article_url(src_files))
    dtype = [ ('idx', 'i'), ('url', np.array(urls).dtype) ]
    context_data = [ np.array(zip(indices, urls), dtype=dtype) ]
    
    if unidecode:
        for i in xrange(len(corpus)):
            if isinstance(corpus[i], unicode):
                corpus[i] = uni.unidecode(corpus[i])

    c = Corpus(corpus, ['document'], context_data)

    return apply_stoplist(c, nltk_stop=nltk_stop,
                          freq=stop_freq, add_stop=add_stop)

