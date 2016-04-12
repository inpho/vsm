import os
import shutil
import re
import logging

import numpy as np

from unidecode import unidecode

from nltk.corpus import wordnet as wn
import enchant

from vsm.extensions.corpusbuilders.util import filter_by_suffix


import json
from time import sleep
from urllib2 import urlopen
from urllib import quote_plus

def metadata(id, sleep_time=1):
    """
    Given a HTRC ID, download the volume metadata from the Solr index.

    
    :param id: HTRC volume id.
    :type id: string

    :param sleep_time: Sleep time to prevent denial of service
    :type sleep_time: int in seconds, default: 1

    :returns: dict

    """
    solr ="http://chinkapin.pti.indiana.edu:9994/solr/meta/select/?q=id:%s" % id
    solr += "&wt=json" ## retrieve JSON results
    # TODO: exception handling
    if sleep_time:
        sleep(sleep_time) ## JUST TO MAKE SURE WE ARE THROTTLED
    try:
        data = json.load(urlopen(solr))
        print id
        return data['response']['docs'][0]
    except ValueError, IndexError:
        print "No result found for " + id 
        return dict()

def proc_htrc_coll(coll_dir, ignore=['.json', '.log', '.err']):
    """
    Given a collection, cleans up plain page files for books in the collection.

    :param coll_dir: The path for collection.
    :type coll_dir: string
    
    :param ignore: List of file extensions to ignore in the directory.
    :type ignore: list of strings, optional

    :returns: None

    :See Also: :meth: proc_htrc_book
    """
    books = os.listdir(coll_dir)
    books = filter_by_suffix(books, ignore)
    
    for book in books:
        # For debugging
        # if book == 'uc2.ark+=13960=t1zc80k1p':
        # if book == 'uc2.ark+=13960=t8tb11c8g':
        # if book == 'uc2.ark+=13960=t74t6gz6r':
        proc_htrc_book(book, coll_dir, ignore=ignore)
            


def proc_htrc_book(book, coll_dir, ignore=['.json', '.log', '.err']):
    """
    Cleans up page headers, line breaks, and hyphens for all plain pages in the book directory. 
    Creates a log file for debugging purposes.  
  
    :param book: The name of the book directory in coll_dir.
    :type book: string
    
    :param coll_dir: The path for collection.
    :type coll_dir: string
    
    :param ignore: List of file extensions to ignore in the directory.
    :type ignore: list of strings, optional

    :returns: None

    :See Also: :meth: rm_pg_headers, :meth: rm_lb_hyphens
    """
    book_root = os.path.join(coll_dir, book)

    logger = logging.getLogger(book)
    logger.setLevel(logging.INFO)
    log_file = os.path.join(coll_dir, book + '-raw-proc.log')
    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    print 'Processing', book_root

    try:
        rm_pg_headers(book_root, logger, ignore=ignore)
        rm_lb_hyphens(book_root, logger, ignore=ignore)
    finally:
        handler.close()



def rm_lb_hyphens(plain_root, logger, ignore=['.json', '.log', '.err']):
    """
    Looks for a hyphen followed by whitespace or a line break.

    Reconstructs word and checks to see if the result exists in either
    WordNet or the OS's default spellchecker dictionary. If so,
    replaces fragments with reconstructed word.
    
    :param plain_root: The name of the directory containing plain-text 
        files.
    :type plain_root: string
    
    :param logger: Logger that handles logging for the given directory.
    :type logger: Logger
    
    :param ignore: List of file extensions to ignore in the directory.
    :type ignore: list of strings, optional

    :returns: None
    """

    try:
        d = enchant.Dict('en_US')
    except ImportError:
        d = None

    def recon(match_obj):
        rc_word = match_obj.group(1) + match_obj.group(2)
        
        if wn.synsets(rc_word) or (d and d.check(rc_word)):
            logger.info('\nbook: %s\nreconstructed word:\n%s\n',
                         plain_root, rc_word)
            return rc_word
        
        logger.info('\nbook: %s\nignored expression:\nleft: %s\nright: %s\n',
                     plain_root, match_obj.group(1), match_obj.group(2))

        return match_obj.group(0)

    def inner(s):
        lb_hyphenated = re.compile(r'(\w+)-\s+(\w+)')
        return lb_hyphenated.sub(recon, s)
    
    page_files = os.listdir(plain_root)
    page_files = filter_by_suffix(page_files, ignore)

    for i, page_file in enumerate(page_files):
        filename = os.path.join(plain_root, page_file)

        with open(filename, 'r+w') as f:
            page = f.read()
            page = inner(page)
            f.seek(0)
            f.write(page)
            f.truncate()



def rm_pg_headers(plain_root, logger, bound=1, ignore=['.json', '.log', '.err']):
    """
    Tries to detect repeated page headers (e.g., chapter titles). If
    found, removes them.

    The routine takes the first non-empty lines of text, strips them
    of numbers and punctuation and computes frequencies. If frequency
    for the reduced string exceeds `bound`, the corresponding first
    lines are considered headers.
    
    :param plain_root: The name of the directory containing plain-text 
        files.
    :type plain_root: string
    
    :param logger: Logger that handles logging for the given directory.
    :type logger: Logger
    
    :param bound: Number of frequency of a reduced string. If the string
        appears more than `bound`, then the corresponding first lines are
        considered headers. Default is 1.
    :param bound: int, optional

    :param ignore: List of file extensions to ignore in the directory.
    :type ignore: list of strings, optional

    :returns: None
    """
    page_files = os.listdir(plain_root)
    page_files = filter_by_suffix(page_files, ignore)

    # Get first non-empty lines
    first_lines = []
    fl = re.compile(r'^\s*([^\n]*)\n')
    
    for page_file in page_files:
        page_file = os.path.join(plain_root, page_file)

        with open(page_file, 'r') as f:
            page = f.read()

        first_line = fl.match(page)
        if first_line == None:
            first_lines.append('')
        else:
            first_lines.append(first_line.group(0))

    # Remove capitalization, roman numerals for numbers under 50,
    # punctuation, arabic numerals from first lines
    for i in xrange(len(first_lines)):
        line = first_lines[i]
        line = line.lower()

        # An overzealous arabic numeral detector (OCR errors include
        # `i` for `1` for example)
        line = re.sub(r'\b\S*\d+\S*\b', '', line)

        # Roman numerals i to xxxix
        line = re.sub(r'\b(x{0,3})(ix|iv|v?i{0,3})\b', '', line)

        # Collapse line to letters only
        line = re.sub(r'[^a-z]', '', line)
        first_lines[i] = (first_lines[i], line)

    freqs = dict()
    for line, reduced in first_lines:
        if reduced in freqs:
            freqs[reduced] += 1
        else:
            freqs[reduced] = 1
    
    for i, page_file in enumerate(page_files):
        filename = os.path.join(plain_root, page_file)
        line, reduced = first_lines[i]

        if freqs[reduced] > bound:
            with open(filename, 'r') as f:
                page = f.read()
            if page:
                logger.info('\nbook: %s\nfile: %s\nremoved header:\n%s\n',
                             plain_root, page_file, line)
            page = fl.sub('', page)

            with open(filename, 'w') as f:
                f.write(page)


def htrc_load_metadata_1315():
    """
    Loads hathitrust metadata for the 1315 volumes.
    """
    import os
    import json

    filename = ('/var/inphosemantics/data/20130101/htrc-anthropomorphism-1315/'
                'htrc-1315-metadata.json')

    with open(filename) as f:
        metadata = json.load(f)

    return metadata


def htrc_load_metadata_86():
    """
    Loads hathitrust metadata for the 86 volumes.
    """
    import os
    import json

    filename = ('/var/inphosemantics/data/20130101/htrc-anthropomorphism-86/'
                'htrc-anthropomorphism-86-metadata.json')

    with open(filename) as f:
        metadata = json.load(f)

    return metadata


def htrc_get_titles(metadata, vol_id):
    """
    Gets titles of the volume given the metadata from a json file
    and volume id.
    """
    try:
        md = metadata[vol_id]
        return md[md.keys()[0]]['titles']

    except KeyError:
        print 'Volume ID not found:', vol_id
        raise


def htrc_label_fn_86(metadata):
    """
    A customized label function for hathitrust 86 volumes.
    It loads the metadata of htrc 86 and returns labels that consist of
    file names which are pages and book titles.

    :param metadata: Strucutred array that has 'file' and 'book_label' field.
        Most likely the output of Corpus.view_metadata('page').
    :type metadata: array

    :returns: An array of labels that consist of file names and book titles.

    :See Also: :meth: Corpus.view_metadata
    """
    md = htrc_load_metadata_86()

    files = metadata['file']
    titles = []
    for v in metadata['book_label']:
        title = unidecode(htrc_get_titles(md, v)[0])
        if len(title) > 15:
            title = title[:15]
        titles.append(title)
    
    labels = ['{0}, {1}'.format(t,f) for (t,f) in zip(titles, files)]
    
    return np.array(labels)



def htrc_label_fn_1315(metadata):
    """
    A customized label function for hathitrust 1315 volumes.
    It loads the metadata of htrc 86 and returns book title labels.

    :param metadata: Strucutred array that has 'book_label' field.
        Most likely the output of Corpus.view_metadata('book').
    :type metadata: array

    :returns: An array of labels that consist of book titles.

    :See Also: :meth: Corpus.view_metadata
    """
    md = htrc_load_metadata_1315()

    titles = []
    for v in metadata['book_label']:
        title = unidecode(htrc_get_titles(md, v)[0])
        if len(title) > 60:
            title = title[:60]
        titles.append(title)
    
    return np.array(titles)


def htrc_find_duplicates(metadata, vol_list):
    """
    Takes metadata and a list of volumes and finds duplicates
    amongst the volumes.

    :param metadata: Dictionary of metadata. Output of
        htrc_load_metadata_86() or htrc_load_metadata_1315()
        is acceptable.
    :type metadata: dictionary

    :param vol_list: List of volumes. An example of a volume is
        'uc2.ark+=13960=t73t9h556'.
    :type vol_list: list

    :returns: indices : the indices of duplicates from `vol_list`.
    
    :See Also: :meth: htrc_load_metadata_86, :meth: htrc_load_metadata_1315
    """
    record_ids = [metadata[vol].keys()[0] for vol in vol_list]
    
    mem, indices = [], []

    for i,r in enumerate(record_ids):
        if r in mem:
            indices.append(i)
        elif r in record_ids[:i]:
            indices.append(i)
            mem.append(r)

    return indices


def add_link(s):
    """
    if `s` is a url, then adds anchor tags for html representation in ipynb.
    """
    if s.startswith('http'):
        a = '<a href="{0}" target="_blank">'.format(s)
        a += s
        a += '</a>'
        return a

def htrc_label_link_fn_86(metadata):
    """
    A customized label function for hathitrust 86 volumes.
    It loads the metadata of htrc 86 and returns pages, book title labels, and page urls
    that open up the page on hathitrust website.

    :param metadata: Strucutred array that has 'file', and 'book_label' field.
        Most likely the output of Corpus.view_metadata('page').
    :type metadata: array

    :returns: An array of labels for pages, books, and page urls.

    :See Also: :meth: Corpus.view_metadata, :meth: htrc_label_fn_86
    """
    md = htrc_load_metadata_86()

    files = metadata['file'] 
    titles = []
    for v in metadata['book_label']:
        title = unidecode(htrc_get_titles(md, v)[0])
        if len(title) > 1:
            title = title[:15]
        titles.append(title)
    
    names = [name for name in metadata.dtype.names if name.endswith('_label')]
    links = [add_link(x[n]) for n in names if n.endswith('_url_label') for x in metadata]

    labels = ['{0}, {1}, {2}'.format(t,f,l) for (t,f,l) in zip(titles, files, links)]
    return np.array(labels)


def htrc_label_link_fn_1315(metadata):
    """
    A customized label function for hathitrust 1315 volumes.
    It loads the metadata of htrc 1315 and returns book title labels and their urls
    that open up the book on hathitrust website.

    :param metadata: Strucutred array that has 'book_label' field.
        Most likely the output of Corpus.view_metadata('page').
    :type metadata: array

    :returns: An array of labels for books and book urls.

    :See Also: :meth: Corpus.view_metadata, :meth: htrc_label_fn_1315
    """
    md = htrc_load_metadata_1315()

    titles = []
    for v in metadata['book_label']:
        title = unidecode(htrc_get_titles(md, v)[0])
        if len(title) > 60:
            title = title[:60]
        titles.append(title)
    
    names = [name for name in metadata.dtype.names if name.endswith('_label')]
    links = [add_link(x[n]) for n in names if n.endswith('_url_label') for x in metadata]

    labels = ['{0}, {1}'.format(t,l) for (t,l) in zip(titles, links)]
    return np.array(labels)



def url_metadata(corpus, ctx_type, coll_dir):
    """
    Returns a list of urls whose order matches with the existing metadata.
    It creates url metadata that can be added to a Corpus object with
    add_metadata function in vsm.ext.corpusbuilders.util.

    :param corpus: Corpus to add url metadata to. Urls match with the existing
        metadata of `corpus`.
    :type corpus: Corpus

    :param ctx_type: A type of tokenization.
    :type ctx_type: string

    :param coll_dir: Path for the collection directory. Either htrc 86 plain
        or htrc 1315 plain directory.
    :type coll_dir: string

    :returns: md : List of urls to be added to corpus

    :See Also: :meth: add_metadata
    """

    import json
    from vsm.viewer import doc_label_name
    import re

    urls = []
    corp_md = corpus.view_metadata('book')
    book_labels = corp_md[doc_label_name('book')]
    
    for book_label in book_labels:
        coll_path = os.path.join(coll_dir, book_label)
        booklist = os.listdir(coll_path)
        book = filter_by_suffix(booklist, ignore=['.txt', '.pickle'])
        
        book_path = os.path.join(coll_path, book[0])
        with open(book_path, 'r') as f:
            d = json.load(f)
            url = ''
            li = sorted(d['items'], key=lambda k: int(k['lastUpdate']))
            url = li[-1]['itemURL']

            if ctx_type == 'book':
                urls.append( unidecode(url))
            else: # urls for pages
                page_md = corpus.view_metadata('page')
                files = [a for a in page_md['file'] if a.startswith(book_label)]
                nums = [re.findall('[1-9][0-9]*', a)[-1] for a in files]
                for i in nums: 
                    s = url + '?urlappend=%3Bseq={0}'.format(i)
                    urls.append( unidecode(s))
    return urls


def page_url(corpus, ctx_type, book_path, book_id, jsonfile):
    """
    Modified htrc_*_label_fn. The individual volumes don't have 'book' as a context type.
    """
    import json
    from vsm.viewer import doc_label_name
    import re

    urls = []
    corp_md = corpus.view_metadata('page')

    jsonpath = os.path.join(book_path, jsonfile)
    with open(jsonpath, 'r') as f:
        md = json.load(f)
        url = ''
        li = sorted(md['items'], key=lambda k: int(k['lastUpdate']))
        url = li[-1]['itemURL']
            
        if ctx_type == 'book':
            urls.append( unidecode(url))
        else: # urls for pages
            page_md = corpus.view_metadata('page')
            files = page_md[doc_label_name('page')] 
            
            nums = [re.findall('[1-9][0-9]*', a)[-1] for a in files]
            for i in nums:
                s = url + '?urlappend=%3Bseq={0}'.format(i)
                urls.append( unidecode(s))
    return urls


def add_link_(s, addee):
    """
    Returns <a href="s">addee</a> 
    For example, <a href="page_url">page_label</a>
    """
    if s.startswith('http'):
        a = '<a href="{0}" target="_blank">'.format(s)
        a += addee
        a += '</a>'
        return a

def vol_link_fn(md): 
    """
    Not a generalized function for individual htrc volume with page links
    and sentence labels. 'sentences_label' is CorpusSent.sentences which
    is an array or original sentences.

    :ref: vsm.extensions.ldasentences CorpusSent
    """
    # md == corpus.view_metadata('sentence')
    links = [add_link_(x['page_urls'], x['page_label']) for x in md]
    labels = ['{0}, {1}, {2}'.format(l, i, s) for (l,i,s) in
             zip(links, md['sentence_label'], md['sentences_label'])]

    return np.array(labels)
