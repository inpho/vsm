import os
import shutil
import re
import logging

import numpy as np

from unidecode import unidecode

from nltk.corpus import wordnet as wn
import enchant

from vsm.corpus.util import filter_by_suffix
from vsm.viewer.ldagibbsviewer import LDAGibbsViewer


def proc_htrc_coll(coll_dir, ignore=['.json', '.log']):
    """
    Given a collection, cleans up plain page files for books in the collection.

    Parameters
    ----------
    coll_dir : string
        The path for collection.
    ignore : list of strings, optional
        List of file extensions to ignore in the directory.

    See Also
    --------
    proc_htrc_book
    """
    books = os.listdir(coll_dir)
    books = filter_by_suffix(books, ignore)
    
    for book in books:
        # For debugging
        # if book == 'uc2.ark+=13960=t1zc80k1p':
        # if book == 'uc2.ark+=13960=t8tb11c8g':
        # if book == 'uc2.ark+=13960=t74t6gz6r':
        proc_htrc_book(book, coll_dir, ignore=ignore)
            


def proc_htrc_book(book, coll_dir, ignore=['.json', '.log']):
    """
    Cleans up page headers, line breaks, and hyphens for all plain pages in the book directory. 
    Creates a log file for debugging purposes.  
  
    Parameters
    ----------
    book : string
        The name of the book directory in coll_dir.
    coll_dir : string
        The path for collection.
    ignore : list of strings, optional
        List of file extensions to ignore in the directory.

    See Also
    --------
    rm_pg_headers
    rm_lb_hyphens
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



def rm_lb_hyphens(plain_root, logger, ignore=['.json', '.log']):
    """
    Looks for a hyphen followed by whitespace or a line break.

    Reconstructs word and checks to see if the result exists in either
    WordNet or the OS's default spellchecker dictionary. If so,
    replaces fragments with reconstructed word.
    """

    d = enchant.Dict('en_US')

    def recon(match_obj):
        rc_word = match_obj.group(1) + match_obj.group(2)
        
        if wn.synsets(rc_word) or d.check(rc_word):
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



def rm_pg_headers(plain_root, logger, bound=1, ignore=['.json', '.log']):
    """
    Tries to detect repeated page headers (e.g., chapter titles). If
    found, removes them.

    The routine takes the first non-empty lines of text, strips them
    of numbers and punctuation and computes frequencies. If frequency
    for the reduced string exceeds `bound`, the corresponding first
    lines are considered headers.
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

    import os
    import json

    filename = ('/var/inphosemantics/data/20130101/htrc-anthropomorphism-1315/'
                'htrc-1315-metadata.json')

    with open(filename) as f:
        metadata = json.load(f)

    return metadata


def htrc_load_metadata_86():

    import os
    import json

    filename = ('/var/inphosemantics/data/20130101/htrc-anthropomorphism-86/'
                'htrc-anthropomorphism-86-metadata.json')

    with open(filename) as f:
        metadata = json.load(f)

    return metadata


def htrc_get_titles(metadata, vol_id):

    try:
        md = metadata[vol_id]
        return md[md.keys()[0]]['titles']

    except KeyError:
        print 'Volume ID not found:', vol_id
        raise


def htrc_label_fn_86(metadata):
    """
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
    """
    if s.startswith('http'):
        a = '<a href="{0}" target="_blank">'.format(s)
        a += s
        a += '</a>'
        return a

def htrc_label_link_fn_86(metadata):
    """
    """
    md = htrc_load_metadata_86()

    titles = []
    for v in metadata['book_label']:
        title = unidecode(htrc_get_titles(md, v)[0])
        if len(title) > 60:
            title = title[:60]
        titles.append(title)
    
    names = [name for name in metadata.dtype.names if name.endswith('_label')]
    links = [add_link(x[n]) for n in names if n.endswith('_url_label') for x in metadata]

    arr = np.array(zip(titles, links), dtype=[('titles', '|S60'), ('links', '|S160')])
    dtypes = ['titles', 'links']
    return np.array([', '.join([x[n] for n in dtypes]) for x in arr])


def htrc_label_link_fn_1315(metadata):
    """
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

    arr = np.array(zip(titles, links), dtype=[('titles', '|S60'), ('links', '|S160')])
    dtypes = ['titles', 'links']
    return np.array([', '.join([x[n] for n in dtypes]) for x in arr])



def url_metadata(corpus, ctx_type, coll_dir):
    """
    Returns a list of urls whose order matches with the existing metadata.
    It creates url metadata that can be added to a Corpus object with
    add_metadata function in vsm.corpus.util.
    """

    import json
    from vsm.viewer import doc_label_name

    md = []
    corp_md = corpus.view_metadata(ctx_type)
    book_labels = corp_md[doc_label_name('book')]

    for book_label in book_labels:
        coll_path = os.path.join(coll_dir, book_label)
        booklist = os.listdir(coll_path)
        book = filter_by_suffix(booklist, ignore=['.txt', '.pickle'])
       
        book_path = os.path.join(coll_path, book[0])
        with open(book_path, 'r') as f:
            d = json.load(f)
            for k in d.keys():
                if k == 'items':
                    li = sorted(d[k], key=lambda k: int(k['lastUpdate']))
                    url = li[-1]['itemURL']

                    if ctx_type == 'page' or ctx_type == 'sentence':
                        for i in xrange(1, len(booklist)-1):
                            s = url + '?urlappend=%3Bseq={0}'.format(i)
                            md.append( unidecode(s) )
                    else:
                        md.append( unidecode(url))
    return md




