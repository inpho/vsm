import corpus
import lda
import preprocess
import postprocess
import experiment


def convert_tokens(vsm_corpus, tok_name):

    hwc_corpus = corpus.Corpus()

    # vsm_corpus = vsm_mcorpus.to_corpus(compress=True)

    articles = vsm_corpus.view_tokens(tok_name, as_strings=True)

    metadata = vsm_corpus.view_metadata(tok_name)

    #TODO: Fix view_metadata so that it always returns a list of strings

    metadata = [str(d) for d in metadata]

    for article, metadata in zip(articles, metadata):

        article = article.tolist()

        hwc_corpus.add(metadata, article)

    return hwc_corpus
