import numpy as np

from vsm.corpus import Corpus
from vsm.corpus.util import *


class CorpusSent(Corpus):
    """
    A subclass of Corpus whose purpose is to store original
    sentence information in the Corpus
    """
    def __init__(self, corpus, sentences, context_types=[], context_data=[], 
		remove_empty=False):
       super(CorpusSent, self).__init__(corpus, context_types=context_types,
		 context_data=context_data, remove_empty=remove_empty)
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

        Parameters
        ----------
	stoplist : list
	    The list of words to be removed.
	freq : integer, optional
	    A threshold where words of frequency <= 'freq' are removed. 
	    Default is 0.
            
        Returns
        -------
	Copy of corpus with words in the stoplist and words of frequnecy
	<= 'freq' removed.

        See Also
        --------
        Corpus
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

        return CorpusSent(corpus, self.sentences, context_data=context_data, context_types=self.context_types)


    @staticmethod
    def load(file):
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
        
        Parameters
        ----------
        file : str-like or file-like object
            Designates the file to which to save data. See
            `numpy.savez` for further details.
            
        Returns
        -------
        None

        See Also
        --------
        Corpus
        Corpus.load
        numpy.savez
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
    sent_label_dt = ('sent_label', np.array(sent_n, np.str_).dtype)
    corpus_data = dict()
    dtype = [idx_dt, label_dt]
    corpus_data[chunk_name] = np.array(chk_tokens, dtype=dtype)

    if paragraphs:
        par_label_dt = ('par_label', np.array(par_n, np.str_).dtype)
        dtype = [idx_dt, label_dt, par_label_dt]
        corpus_data['paragraph'] = np.array(par_tokens, dtype=dtype)
        dtype = [idx_dt, label_dt, par_label_dt, sent_label_dt]
        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)
    else:
        dtype = [idx_dt, label_dt, sent_label_dt]
        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    return words, corpus_data, sent_orig



def dir_corpus(plain_dir, chunk_name='article', paragraphs=True,
               nltk_stop=True, stop_freq=1, add_stop=None, corpus_sent=False):
    """
    `dir_corpus` is a convenience function for generating Corpus
    objects from a directory of plain text files.

    `dir_corpus` will retain file-level tokenization and perform
    sentence and word tokenizations. Optionally, it will provide
    paragraph-level tokenizations.

    It will also strip punctuation and arabic numerals outside the
    range 1-29. All letters are made lowercase.

    Parameters
    ----------
    plain_dir : string-like
        String containing directory containing a plain-text corpus.
    chunk_name : string-line
        The name of the tokenization corresponding to individual
        files. For example, if the files are pages of a book, one
        might set `chunk_name` to `pages`. Default is `articles`.
    paragraphs : boolean
        If `True`, a paragraph-level tokenization is included.
        Defaults to `True`.
    nltk_stop : boolean
        If `True` then the corpus object is masked using the NLTK
        English stop words. Default is `False`.
    stop_freq : int
        The upper bound for a word to be masked on the basis of its
        collection frequency. Default is 0.
    add_stop : array-like
        A list of stop words. Default is `None`.

    Returns
    -------
    c : a Corpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.

    """
    chunks = []
    filenames = os.listdir(plain_dir)
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
    c = apply_stoplist(c, nltk_stop=nltk_stop,
                       freq=stop_freq, add_stop=add_stop)

    return c
