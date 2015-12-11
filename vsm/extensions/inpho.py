from inpho.model import *

ideas = Session.query(Idea)
words_int = dict([(idea.label, idea.ID) for idea in ideas.all()])

def inpho_word_tokenize(document, terms=None):
    if terms is None:
        terms = ideas[:]
    occurrences = []
    
    # iterate over terms to be scanned
    for term in terms:
        # build list of search patterns starting with label
        for pattern in term.patterns:
            try:
                if re.search(pattern, document, flags=re.IGNORECASE):
                    occurrences.append(str(term.ID))
                    break
            except re.error:
                logging.warning('Term %d (%s) pattern "%s" failed' % 
                                (term.ID, term.label, pattern))

    return occurrences


