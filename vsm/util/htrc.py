import os
import shutil
import re
import logging

from nltk.corpus import wordnet as wn
import enchant

from corpustools import filter_by_suffix

__all__ = ['proc_htrc_coll']



def proc_htrc_coll(coll_dir, ignore=['.json', '.log']):
    """
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
