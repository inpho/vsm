class Alphabet(object):

    def __init__(self):

        self._mapping = {} # mapping from strings to integers
        self._reverse = {} # reverse mapping from integers to strings

        self._idx = 0
        self._growing = True

    def stop_growth(self):
        self._growing = False

    def lookup(self, i):

        assert isinstance(i, int)
        return self._reverse[i]

    def plaintext(self):

        contents = self._reverse.items()
        contents.sort(key=lambda x: x[0])

        return '\n'.join('%s\t%s' % (i, s) for i, s in contents)

    def __contains__(self, s):

        assert isinstance(s, basestring)
        return s in self._mapping

    def __getitem__(self, s):

        try:
            return self._mapping[s]
        except KeyError:
            if not isinstance(s, basestring):
                raise ValueError('Invalid key (%s): must be a string.' % (s,))
            if not self._growing:
                return None
            i = self._mapping[s] = self._idx
            self._reverse[i] = s
            self._idx += 1
            return i

    add = __getitem__

    def __iter__(self):

        for i in xrange(len(self)):
            yield self._reverse[i]

    def __len__(self):
        return len(self._mapping)
