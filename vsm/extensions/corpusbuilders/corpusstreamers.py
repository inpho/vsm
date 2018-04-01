from concurrent.futures import as_completed, ProcessPoolExecutor
import pickle
import tempfile
import os

from progressbar import ProgressBar, Bar, Percentage

from vsm.extensions.corpusbuilders import corpus_fromlist
from vsm.extensions.corpusbuilders.util import apply_stoplist, word_tokenize

IGNORE = ['.json','.log','.pickle', '.DS_Store']

def tokenize_and_pickle_file(filename, pickle_dir=None, tokenizer=word_tokenize):
    """
    Tokenizes a file and returns a filename of a PickledWords instance.
    """
    with open(filename) as infile:
        data = infile.read()

    corpus = word_tokenize(data)

    # dump to picklefile
    with tempfile.NamedTemporaryFile(dir=pickle_dir, delete=False) as fp:
        pickle.dump(corpus, fp)
        filename = fp.name
    del corpus

    return filename


def corpus_from_files(filenames, encoding='utf8', ignore=IGNORE,
    nltk_stop=False, stop_freq=0, add_stop=None, decode=False, 
    verbose=True, simple=False, tokenizer=word_tokenize):
    if os.path.isdir(filenames):
        filenames = [os.path.join(filenames, p) for p in os.listdir(filenames)]

    if verbose:
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(filenames))
        pbar = pbar.start()
        n = 0

    with tempfile.TemporaryDirectory(prefix='vsm-') as pickle_dir:
        with ProcessPoolExecutor() as executor:
            corpus = [executor.submit(tokenize_and_pickle_file, filename, pickle_dir, tokenizer)
                          for filename in filenames]
            if verbose:
                for f in as_completed(corpus):
                    n += 1
                    pbar.update(n)

            pbar.finish()
            corpus = [f.result() for f in corpus]
        
        corpus = [PickledWords(f) for f in corpus]

        corpus = corpus_fromlist(corpus, context_type='document')

    corpus = apply_stoplist(corpus, nltk_stop=nltk_stop, freq=stop_freq)

    return corpus


class PickledWords:
    def __init__(self, filename):
        self.file = filename

        with open(self.file, 'rb') as fp:
            self.list = pickle.load(fp)
            self.len = len(self.list)
            del self.list
    
    def __iter__(self):
        with open(self.file, 'rb') as fp:
            self.list = pickle.load(fp)
    
        for i in range(len(self.list)):
            yield self.list[i]

        del self.list

        raise StopIteration()

    def __len__(self):
        return self.len

    def __copy__(self):
        return PickledWords(self.file)