import os
import re

import numpy as np

import nltk



def strip_punc(tsent):

    p1 = re.compile(r'^(\W*)')

    p2 = re.compile(r'(\W*)$')

    out = []

    for word in tsent:

        w = re.sub(p2, '', re.sub(p1, '', word))

        if w:

            out.append(w)

    return out



def rem_num(tsent):

    p = re.compile(r'(^\D+$)|(^\D*[1-2]\d\D*$|^\D*\d\D*$)')

    return [word for word in tsent if re.search(p, word)]



def rehyph(sent):

    return re.sub(r'(?P<x1>.)--(?P<x2>.)', '\g<x1> - \g<x2>', sent)



def mask_corpus(c, nltk_stop=False, mask_freq=0, add_stop=None):

    from vsm.corpus import mask_from_stoplist, mask_freq_t

    stoplist = set()

    if nltk_stop:

        for w in nltk.corpus.stopwords.words('english'):

            stoplist.add(w)

    if add_stop:

        for w in add_stop:

            stoplist.add(w)
            
    if stoplist:

        mask_from_stoplist(c, list(stoplist))

    if mask_freq > 0:

        mask_freq_t(c, mask_freq)

    return c



def filter_by_suffix(l, ignore):

    return [e for e in l if not sum([e.endswith(s) for s in ignore])]



