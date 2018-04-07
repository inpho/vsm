from finalcorpus import *
import gzip

start_idx = 0
m = lda_m[20]
metadata = c.view_metadata(m.context_type)

def export_model():
    with gzip.open('model_to_mallet.gz', 'wb') as f:
        f.write("#doc source pos typeindex type topic")
        alpha = m.alpha
        f.write(alpha)
        beta = m.beta

        for end_idx, doc in metadata:
            for i in range(start_idx, end_idx):
                doc = doc
                source = "/"
                pos = i
                typeIndex = c.corpus[i]
                ttype = c.words[c.corpus[i]]
                topic = m.Z[i]
                line = "{} {} {} {} {} {}\n".format(doc, source, pos, typeIndex, ttype, topic)
                f.write(line)
                start_idx = end_idx


def import_model():
    startPos = []
    corpus = []
    z = []
    words = {}
    prevDoc = 0;

    with gzip.open('topic-state.gz', 'rb') as f:
        for i, line in enumerate(f, start = -3):
            #skip first three lines with header info
            if i >= 0:
                #columns - #doc source pos typeindex type topic
                doc, _, _, typeindex, type, topic = line.split() 
                corpus.append(typeindex)
                z.append(topic)
                words[typeindex] = type
                if doc != prevDoc:
                    startPos.append(i)
                prevDoc = doc

