import numpy as np
from vsm.corpus import Corpus
from vsm.extensions.corpusbuilders import *
from vsm.extensions.corpuscleanup import apply_stoplist_len
from vsm.extensions.htrc import vol_link_fn, add_link_
import os
import re

__all__ = ['CorpusSent', 'sim_sent_sent', 'sim_sent_sent_across',
        'file_tokenize', 'file_corpus', 'dir_tokenize', 'dir_corpus',
        'extend_sdd', 'extend_across']


class CorpusSent(Corpus):
    """
    A subclass of Corpus whose purpose is to store original
    sentence information in the Corpus
    
    :See Also: :class: Corpus
    """
    def __init__(self, corpus, sentences, context_types=[], context_data=[], 
		remove_empty=False):
       
       super(CorpusSent, self).__init__(corpus, context_types=context_types,
		 context_data=context_data, remove_empty=remove_empty)
       
       sentences = [re.sub('\n', ' ', s) for s in sentences]
       self.sentences = np.array(sentences)


    def __set_words_int(self):
        """
        Mapping of words to their integer representations.
        """
        self.words_int = dict((t,i) for i,t in enumerate(self.words))


    def apply_stoplist(self, stoplist=[], freq=0):
        """ 
        Takes a Corpus object and returns a copy of it with words in the
        stoplist removed and with words of frequency <= `freq` removed.
        
	    :param stoplist: The list of words to be removed.
        :type stoplist: list
        
        :type freq: integer, optional
	    :param freq: A threshold where words of frequency <= 'freq' are 
            removed. Default is 0.
            
        :returns: Copy of corpus with words in the stoplist and words of
            frequnecy <= 'freq' removed.

        :See Also: :class:`Corpus`
        """
        if freq:
            #TODO: Use the TF model instead

            print 'Computing collection frequencies'
            cfs = np.zeros_like(self.words, dtype=self.corpus.dtype)
    
            for word in self.corpus:
                cfs[word] += 1

            print 'Selecting words of frequency <=', freq
            freq_stop = np.arange(cfs.size)[(cfs <= freq)]
            stop = set(freq_stop)
        else:
            stop = set()

        for t in stoplist:
            if t in self.words:
                stop.add(self.words_int[t])

        if not stop:
            print 'Stop list is empty.'
            return self
    
        print 'Removing stop words'
        f = np.vectorize(lambda x: x not in stop)
        corpus = self.corpus[f(self.corpus)]

        print 'Rebuilding corpus'
        corpus = [self.words[i] for i in corpus]
        context_data = []
        for i in xrange(len(self.context_data)):
            print 'Recomputing token breaks:', self.context_types[i]
            tokens = self.view_contexts(self.context_types[i])
            spans = [t[f(t)].size for t in tokens]
            tok = self.context_data[i].copy()
            tok['idx'] = np.cumsum(spans)
            context_data.append(tok)

        return CorpusSent(corpus, self.sentences, context_data=context_data,
                            context_types=self.context_types)


    @staticmethod
    def load(file):
        """
        Loads data into a Corpus object that has been stored using
        `save`.
        
        :param file: Designates the file to read. If `file` is a string ending
            in `.gz`, the file is first gunzipped. See `numpy.load`
            for further details.
        :type file: string-like or file-like object

        :returns: c : A Corpus object storing the data found in `file`.

        :See Also: :class: Corpus, :meth: Corpus.load, :meth: numpy.load
        """
        print 'Loading corpus from', file
        arrays_in = np.load(file)

        c = CorpusSent([], [])
        c.corpus = arrays_in['corpus']
        c.words = arrays_in['words']
        c.sentences = arrays_in['sentences']
        c.context_types = arrays_in['context_types'].tolist()

        c.context_data = list()
        for n in c.context_types:
            t = arrays_in['context_data_' + n]
            c.context_data.append(t)

        c.__set_words_int()

        return c

    def save(self, file):
        """
        Saves data from a CorpusSent object as an `npz` file.
        
        :param file: Designates the file to which to save data. See
            `numpy.savez` for further details.
        :type file: str-like or file-like object
            
        :returns: None

        :See Also: :class: Corpus, :meth: Corpus.save, :meth: numpy.savez
        """
	
	print 'Saving corpus as', file
        arrays_out = dict()
        arrays_out['corpus'] = self.corpus
        arrays_out['words'] = self.words
        arrays_out['sentences'] = self.sentences
        arrays_out['context_types'] = np.asarray(self.context_types)

        for i,t in enumerate(self.context_data):
            key = 'context_data_' + self.context_types[i]
            arrays_out[key] = t

        np.savez(file, **arrays_out)
        
    
    def sent_int(self, sent):
        """
        sent : list of strings
            List of sentence tokenization.
            The list could be a subset of existing sentence tokenization.
        """
        tok = self.view_contexts('sentence', as_strings=True)
        sent_li = []
        for i in xrange(len(tok)):
            sent_li.append(sent)
        keys = [i for i in xrange(len(tok)) 
                if set(sent_li[i]).issubset(tok[i].tolist())]
        
        n = len(keys)
        if n == 0:
            raise Exception('No token fits {0}.'.format(sent))
        elif n > 1:
            return keys
        return keys[0] 


def sim_sent_sent(ldaviewer, sent, print_len=10):
    """
    ldaviewer : ldaviewer object
    sent : sentence index or sentence as a list of words

    Returns
    -------
    sim_sents : numpy array
        (sentence index, probability) as (i, value) pair.
    tokenized_sents : list of arrays
        List containing tokenized sentences as arrays.
    orig_sents : list of strings
        List containing original sentences as strings.
    """
    from vsm.viewer.ldagibbsviewer import LDAGibbsViewer

    corp = ldaviewer.corpus
    ind = sent
    if isinstance(sent, list) and isinstance(sent[0], str):
        ind = corp.sent_int(sent)
    sim_sents = ldaviewer.sim_doc_doc(ind, print_len=print_len)
    lc = sim_sents['doc'][:print_len]
    lc = [s.split(', ') for s in lc]
    lc = [int(s[-1]) for s in lc]
    
    # only returns print_len length
    tokenized_sents, orig_sents = [], []
    for i in lc:
        tokenized_sents.append(corp.view_contexts('sentence', as_strings=True)[i])
        orig_sents.append(corp.sentences[i])

    return tokenized_sents, orig_sents, sim_sents


def sim_sent_sent_across(ldavFrom, ldavTo, beagleviewer, sent, print_len=10,
                         label_fn=vol_link_fn):
    """
    ldavFrom : ldaviewer object where the sentence is from.
    ldavTo : ldaviewer object to find similar sentences
    beagleviewer : beagleviewer object is used to find
        similar words for words that don't exist in a different corpus.
    sent : sentence index of the corpus that corresponds to ldavFrom,
        or sentence as a list of words

    Returns
    -------
    sim_sents : numpy array
        (sentence index, probability) as (i, value) pair.
    tokenized_sents : list of arrays
        List containing tokenized sentences as arrays.
    orig_sents : list of strings
        List containing original sentences as strings.
    """
    from vsm.viewer.ldagibbsviewer import LDAGibbsViewer
    from vsm.viewer.beagleviewer import BeagleViewer

    def first_in_corp(corp, wordlist):
        """
        Goes down the list to find a word that's in `corp`. 
        Assumes there is a word in the `wordlist` that's in `corp`.
        """
        for w in wordlist:
            if w in corp.words:
                return w

    corp = ldavFrom.corpus # to get sent ind
    ind = sent
    word_list = []
    if isinstance(sent, list) and isinstance(sent[0], str):
        ind = corp.sent_int(sent)
        word_list = sent
    elif isinstance(sent, list):
        word_list = set()
        for i in sent:
            li = set(ldavFrom.corpus.view_contexts('sentence', 
                    as_strings=True)[i])
            word_list.update(li)       

    else: # if sent is an int index
        word_list = ldavFrom.corpus.view_contexts('sentence',
                    as_strings=True)[ind].tolist()

    word_list = list(word_list)
    # Before trying ldavTo.sim_word_word, make sure all words
    # in the list exist in ldavTo.corpus.
    wl = []
    for w in word_list:
        if w not in ldavTo.corpus.words:
            words = beagleviewer.sim_word_word(w)['word']
            replacement = first_in_corp(ldavTo.corpus, words)
            wl.append(replacement)
            print 'BEAGLE composite model replaced {0} by {1}'.format(w, 
                                                        replacement)
        else:
            wl.append(w)
    
    # from ldavFrom:sent -> ldavTo:topics -> ldavTo:sent(doc)
    tops = ldavTo.sim_word_top(wl).first_cols[:(ldavTo.model.K/6)]
    tops = [int(t) for t in tops]
    print "Related topics: ", tops 
    # sim_sents = ldavTo.sim_top_doc(tops, print_len=print_len, 
    #                                as_strings=False)
    # lc = sim_sents['i'][:print_len]
    # tokenized_sents, orig_sents = [], []
    # for i in lc:
    #    tokenized_sents.append(ldavTo.corpus.view_contexts('sentence', as_strings=True)[i])
    #   orig_sents.append(ldavTo.corpus.sentences[i])
    sim_sents = ldavTo.sim_top_doc(tops, print_len=print_len,
                                    label_fn=label_fn)
    return sim_sents


def extend_sdd(args, v, print_len=10):
    """
    Extend table resulting from sim_doc_doc with 
    label_fn = vol_link_fn. Adds an ArgumentMap column.
    """
    from vsm.viewer.ldagibbsviewer import LDAGibbsViewer

    sdd = v.sim_doc_doc(args, label_fn=vol_link_fn, print_len=print_len)
    table_str = sdd._repr_html_()
    rows = table_str.split('</tr>') 

    rows[0] = re.sub("2", "3", rows[0]) + '</tr>'
    rows[1] += '<th style="text-align: center; background: #EFF2FB;">Argument\
                Map</th></tr>'
    
    for i in xrange(2,len(rows)-1):
        a = rows[i].split('</a>, ')
        arg = a[1].split(',')[0]

        arg_map = find_arg(arg) 
        rows[i] += '<td>{0}</td></tr>'.format(arg_map)

    return ''.join(rows)


def extend_across(vFrom, vTo, beagle_v, args, txtFrom, txtTo, print_len=10):
    """
    Extend table resulting from sim_sent_sent_across with
    the label_fn= vol_link_fn. Adds ArgumentMap and Novelty columns.
    """
    from vsm.extensions.htrc import add_link_

    across = sim_sent_sent_across(vFrom, vTo, beagle_v, args, print_len=print_len)
    table_str = across._repr_html_()
    rows = table_str.split('</tr>') 

    rows[0] = re.sub("2", "4", rows[0]) + '</tr>'
    rows[1] += '<th style="text-align: center; background: #EFF2FB;">\
                Argument Map</th><th style="text-align: center; background: \
                #EFF2FB;">Novelty</th></tr>'
    
    for i in xrange(2,len(rows)-1):
        a = rows[i].split('</a>, ')
        arg = a[1].split(',')[0]

        novelty = in_ed1(arg, txtTo, txtFrom)
        arg_map = find_arg(novelty)
       
        # add link to novelty when it's found in the corpusFrom.
        if not novelty == 'new':
            li = novelty.split(' ')
            idx = int(li[0])
            md = vFrom.corpus.view_metadata('sentence')[idx]
            link = add_link_(md['page_urls'], md['sentence_label'])
            li[0] = link
            novelty = ' '.join(li)

        rows[i] += '<td>{0}</td><td>{1}</td></tr>'.format(arg_map, novelty)

    return ''.join(rows)

  
def in_ed1(idx, difftxt, ed1txt):
    """
    Only for sim_sent_sent_across.
    Return ind from ed1txt if i has a equal match.
    Else return 'i'th entry in difftxt.
    
    """
    path = '/var/inphosemantics/data/20131214/Washburn/vsm-data/'
    
    with open(path + ed1txt, 'r') as f1:
        ed1 = f1.read()
        ed1 = ed1.split(',') 
        
        with open(path + difftxt, 'r') as f:
            txt = f.read() 
            entries = txt.split(',')
            
            for i in xrange(len(entries)): 
                if entries[i].startswith(str(idx) + ' '):
                    if '=' in entries[i]:
                        return ed1[i]
                    else:
                        prob = entries[i].split(' ')[1]
                        return ed1[i] + ' ' + prob
            # didn't find idx in the table.
            return 'new'

def find_arg(i):
    """
    Find the arg (e.g. '422')  if i is one of the analyzed args,
    otherwise return ''
    """
    import json

    path = '/var/inphosemantics/data/20131214/Washburn/vsm-data/'
    
    if i == 'new' or '(' in i:
        return ''

    i = int(i)
    with open(path + 'arg_indices.json', 'r') as jsonf:
        indices = json.load(jsonf)
        
        for k in indices:
            if i in indices[k]:
                return str(k)
        return ''


def file_tokenize(text):
    """
    `file_tokenize` is a helper function for :meth:`file_corpus`.
    
    Takes a string that is content in a file and returns words
    and corpus data.

    :param text: Content in a plain text file.
    :type text: string

    :returns: words : List of words.
        Words in the `text` tokenized by :meth:`vsm.corpus.util.word_tokenize`.
        corpus_data : Dictionary with context type as keys and
        corresponding tokenizations as values. The tokenizations
        are np.arrays.
    """
    words, par_tokens, sent_tokens, sent_orig = [], [], [], []
    sent_break, par_n, sent_n = 0, 0, 0

    pars = paragraph_tokenize(text)

    for par in pars:
        sents = sentence_tokenize(par)

        for sent in sents:
            w = word_tokenize(sent)
            words.extend(w)
            sent_break += len(w)
            sent_tokens.append((sent_break, par_n, sent_n))
            sent_orig.append(sent)
            sent_n += 1

        par_tokens.append((sent_break, par_n))
        par_n += 1

    idx_dt = ('idx', np.int32)
    sent_label_dt = ('sentence_label', np.array(sent_n, np.str_).dtype)
    par_label_dt = ('paragraph_label', np.array(par_n, np.str_).dtype)

    corpus_data = dict()
    dtype = [idx_dt, par_label_dt]
    corpus_data['paragraph'] = np.array(par_tokens, dtype=dtype)
    dtype = [idx_dt, par_label_dt, sent_label_dt]
    corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    return words, corpus_data, sent_orig


def file_corpus(filename, nltk_stop=True, stop_freq=1, add_stop=None):
    """
    `file_corpus` is a convenience function for generating Corpus
    objects from a a plain text corpus contained in a single string
    `file_corpus` will strip punctuation and arabic numerals outside
    the range 1-29. All letters are made lowercase.

    :param filename: File name of the plain text file.
    :type plain_dir: string-like
    
    :param nltk_stop: If `True` then the corpus object is masked 
        using the NLTK English stop words. Default is `False`.
    :type nltk_stop: boolean, optional
    
    :param stop_freq: The upper bound for a word to be masked on 
        the basis of its collection frequency. Default is 1.
    :type stop_freq: int, optional
    
    :param add_stop: A list of stop words. Default is `None`.
    :type add_stop: array-like, optional

    :returns: c : a Corpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.
    
    :See Also: :class:`vsm.corpus.Corpus`, 
        :meth:`file_tokenize`, 
        :meth:`vsm.corpus.util.apply_stoplist`
    """
    with open(filename, mode='r') as f:
        text = f.read()

    words, tok, sent = file_tokenize(text)
    names, data = zip(*tok.items())
    
    c = CorpusSent(words, sent, context_data=data, context_types=names,
                    remove_empty=False)
    c = apply_stoplist(c, nltk_stop=nltk_stop,
                       freq=stop_freq, add_stop=add_stop)

    return c



def dir_tokenize(chunks, labels, chunk_name='article', paragraphs=True):
    """
    """
    words, chk_tokens, sent_tokens, sent_orig = [], [], [], []
    sent_break, chk_n, sent_n = 0, 0, 0

    if paragraphs:
        par_tokens = []
        par_n = 0
        
        for chk, label in zip(chunks, labels):
            print 'Tokenizing', label
            pars = paragraph_tokenize(chk)

            for par in pars:
                sents = sentence_tokenize(par)

                for sent in sents:
                    w = word_tokenize(sent)
                    words.extend(w)
                    sent_break += len(w)
                    sent_tokens.append((sent_break, label, par_n, sent_n))
                    sent_orig.append(sent)
                    sent_n += 1

                par_tokens.append((sent_break, label, par_n))
                par_n += 1

            chk_tokens.append((sent_break, label))
            chk_n += 1
    else:
        for chk, label in zip(chunks, labels):
            print 'Tokenizing', label
            sents = sentence_tokenize(chk)

            for sent in sents:
                w = word_tokenize(sent)
                words.extend(w)
                sent_break += len(w)
                sent_tokens.append((sent_break, label, sent_n))
                sent_orig.append(sent)
                sent_n += 1

            chk_tokens.append((sent_break, label))
            chk_n += 1

    idx_dt = ('idx', np.int32)
    label_dt = (chunk_name + '_label', np.array(labels).dtype)
    sent_label_dt = ('sentence_label', np.array(sent_n, np.str_).dtype)
    corpus_data = dict()
    dtype = [idx_dt, label_dt]
    corpus_data[chunk_name] = np.array(chk_tokens, dtype=dtype)

    if paragraphs:
        par_label_dt = ('paragraph_label', np.array(par_n, np.str_).dtype)
        dtype = [idx_dt, label_dt, par_label_dt]
        corpus_data['paragraph'] = np.array(par_tokens, dtype=dtype)
        dtype = [idx_dt, label_dt, par_label_dt, sent_label_dt]
        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)
    else:
        dtype = [idx_dt, label_dt, sent_label_dt]
        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    return words, corpus_data, sent_orig



def dir_corpus(plain_dir, chunk_name='article', paragraphs=True, word_len=2,
               nltk_stop=True, stop_freq=1, add_stop=None, corpus_sent=True,
               ignore=['.log', '.pickle', '.xml']):
    """
    `dir_corpus` is a convenience function for generating Corpus
    objects from a directory of plain text files.

    `dir_corpus` will retain file-level tokenization and perform
    sentence and word tokenizations. Optionally, it will provide
    paragraph-level tokenizations.

    It will also strip punctuation and arabic numerals outside the
    range 1-29. All letters are made lowercase.

    :param plain_dir: String containing directory containing a 
        plain-text corpus.
    :type plain_dir: string-like
    
    :param chunk_name: The name of the tokenization corresponding 
        to individual files. For example, if the files are pages 
        of a book, one might set `chunk_name` to `pages`. Default 
        is `articles`.
    :type chunk_name: string-like, optional
    
    :param paragraphs: If `True`, a paragraph-level tokenization 
        is included. Defaults to `True`.
    :type paragraphs: boolean, optional
    
    :param word_len: Filters words whose lengths are <= word_len.
        Default is 2.
    :type word_len: int, optional

    :param nltk_stop: If `True` then the corpus object is masked 
        using the NLTK English stop words. Default is `False`.
    :type nltk_stop: boolean, optional
    
    :param stop_freq: The upper bound for a word to be masked on 
        the basis of its collection frequency. Default is 1.
    :type stop_freq: int, optional

    :param corpus_sent: If `True` a CorpusSent object is returned.
        Otherwise Corpus object is returned. Default is `True`. 
    :type corpus_sent: boolean, optional

    :param add_stop: A list of stop words. Default is `None`.
    :type add_stop: array-like, optional

    :param ignore: The list containing suffixes of files to be filtered.
        The suffix strings are normally file types. Default is ['.json',
        '.log', '.pickle'].
    :type ignore: list of strings, optional

    :returns: c : Corpus or CorpusSent
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.
    
    :See Also: :class: Corpus, :class: CorpusSent, :meth: dir_tokenize,
        :meth: apply_stoplist
    """
    chunks = []
    filenames = os.listdir(plain_dir)
    filenames = filter_by_suffix(filenames, ignore)
    filenames.sort()

    for filename in filenames:
        filename = os.path.join(plain_dir, filename)
        with open(filename, mode='r') as f:
            chunks.append(f.read())

    words, tok, sent = dir_tokenize(chunks, filenames, chunk_name=chunk_name,
                              paragraphs=paragraphs)
    names, data = zip(*tok.items())
    
    if corpus_sent:
        c = CorpusSent(words, sent, context_data=data, context_types=names,
			remove_empty=False)
    else:
        c = Corpus(words, context_data=data, context_types=names)
    c = apply_stoplist_len(c, nltk_stop=nltk_stop, add_stop=add_stop,
                       word_len=word_len, freq=stop_freq)

    return c
