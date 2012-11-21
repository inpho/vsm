import argparse, csv, re
from corpus import *
from numpy import *

def create_stopword_list(stopword_file):

    stopwords = []

    for word in open(stopword_file):
        stopwords.append(word.strip())

    return set(stopwords)

def remove_stopwords(data, stopwords):

    assert isinstance(data, list)
    assert isinstance(stopwords, set)

    return [x for x in data if x not in stopwords]

def tokenize(data):

    assert isinstance(data, basestring)
    return re.findall('\w+', data.lower())

def main():

    # parse command-line arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', metavar='input-file', help='CSV file to be preprocessed')
    parser.add_argument('--remove-stopwords', metavar='stopword-file', help='remove stopwords provided in the specified file')
    parser.add_argument('--output-file', metavar='output-file', help='save preprocessed data to the specified file')

    args = parser.parse_args()

    # create stopword list

    if args.remove_stopwords != None:
        stopwords = create_stopword_list(args.remove_stopwords)

    # preprocess data

    corpus = Corpus()

    for name, label, data in csv.reader(open(args.input_file), delimiter='\t'):
        if args.remove_stopwords != None:
            corpus.add(name, remove_stopwords(tokenize(data), stopwords))
        else:
            corpus.add(name, tokenize(data))

    print '# documents =', len(corpus)
    print '# tokens =', sum(len(doc) for doc in corpus)
    print '# unique types =', len(corpus.alphabet)

    if args.output_file:
        corpus.save(args.output_file)

if __name__ == '__main__':
    main()
