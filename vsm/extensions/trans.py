#!/usr/bin/env python

import nltk
import re
import os
from  unidecode import unidecode
from translate import Translator as Ts
from vsm.corpus.util import *

"""
Uses `translate` python module from https://pypi.python.org/pypi/translate
"""

def sent_tokenize(text, lang='english'):
    tokenizer = nltk.data.load('tokenizers/punkt/{0}.pickle'.format(lang))
    return tokenizer.tokenize(text)

def cleanup(s):
    s = rehyph(s)
    s = s.strip('\n')
    def replace(match):
        if match:
            if match.group(0).startswith(r'\*'):
                return unidecode(match.group(0))
        else:
            return ''

    return re.sub(r"[\x90-\xff]", replace, s)


def transwrapper(text, from_lang, to_lang):
    
    if from_lang == 'en':
        lang = 'english'
    elif from_lang == 'fr':
        lang = 'french'
    elif from_lang == 'de':
        lang = 'german'
    sli = sent_tokenize(text, lang=lang)
    
    out = ''
    for sent in sli:
        sent = cleanup(sent) 
        
        ts = Ts(from_lang=from_lang, to_lang=to_lang)
        target = ts.translate(sent)
        out += target
    
    return out


if __name__=="__main__":
    frompath = 'darwin-de/'
    topath = 'darwin-de-translate/'
    
    books = os.listdir(frompath)
    books.sort()
    
    for book in books:
        book_path = os.path.join(frompath, book)
        print book_path
        pages = os.listdir(book_path)
        pages.sort()
        
        for page in pages:
            page_name = os.path.join(book_path, page)
            
            with open(page_name, 'r') as f:
                try:
                    out = transwrapper(f.read(), 'de', 'en')
                    out = out.encode('utf-8')
                except:
                    out = ''
                    print page_name, ' failed translation.'
                
                try:
                    os.mkdir(os.path.join(topath, book))
                except OSError:
                    pass
                topage = os.path.join(topath, book, page)
                with open(topage, 'w') as fout:
    """ 
    # for individual pages 
    fin = 'darwin-de/wu.89101307601/00000636.txt'
    fout = 'darwin-de-translate/wu.89101307601/00000636.txt'

    with open(fin, 'r') as f:
        out = transwrapper(f.read(), 'de', 'en')
        out = out.encode('utf-8')
        with open(fout, 'w') as fo:
            fo.write(out)
                fout.write(out)"""
