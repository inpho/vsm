import os
import json

root = '/var/inphosemantics/data/20130101/htrc-anthropomorphism-86/plain/'

f = os.listdir(root)

f = [n for n in f if not n.endswith('.log')]

metadata = dict()

for b in f:
    p = os.listdir(root + b)
    s = [n for n in p if n.endswith('.json')]

    if len(s) != 1:
        raise Exception(len(s) + ' json files found in ' + b)

    filename = root + b + '/' + s[0]

    with open(filename, 'r') as fl:
        md = json.load(fl)

        # if md['records'].keys()[0] in metadata:

        #     print metadata[md['records'].keys()[0]]

        #     print md['records']

        #     print b

        #     raise Exception('metadata already submitted for '
        #                     + md['records'].keys()[0])

        metadata.update({ b: md['records'] })

with open('/var/inphosemantics/data/20130101/htrc-anthropomorphism-86/'
          'htrc-anthropomorphism-86-metadata.json', 'w') as fl:

    json.dump(metadata, fl)
