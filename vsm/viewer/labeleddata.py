import numpy as np

from vsm import enum_sort, map_strarr, isfloat, isint, isstr



def format_entry(x):
    """
    """
    # np.void is the type of the tuples that appear in numpy
    # structured arrays
    if isinstance(x, np.void):
        return ', '.join([format_entry(i) for i in x.tolist()]) 
        
    if isfloat(x):
        return '{0:.5f}'.format(x)

    return str(x)


class LabeledColumn(np.ndarray):
    """
    A subclass of nd.ndarray whose purpose is to store labels and
    formatting information for a 1-dimensional structured array. It
    also provides pretty-printing routines.

    A column can have a header and a default display length.

    A subcolumn wraps the data found under a given field name. Each
    subcolumn has a label and a display width.
    """
    def __new__(cls, input_array):
        """
        """
        obj = np.asarray(input_array).view(cls)
        # obj.str_len = None

        return obj


    def __array_finalize__(self, obj):
        """
        """
        if obj is None: return

        # self.str_len = getattr(obj, 'str_len', None)


    # def __str__(self):

    #     pass


class DataTable(np.ndarray):
    """
    A subclass of nd.ndarray whose purpose is to store labels and
    formatting information for an array of LabelColumns. It also
    provides pretty-printing routines.

    A table has a main header and a default display width. If the
    total width of LabeledColumns exceeds this bound, the table is
    split into a sequence of chunks which are displayed accordingly.
    """
    def __new__(cls, input_array):
        """
        """
        obj = np.asarray(input_array).view(cls)
        # obj.str_len = None

        return obj


    def __array_finalize__(self, obj):
        """
        """
        if obj is None: return

        # self.str_len = getattr(obj, 'str_len', None)


    # def __str__(self):

    #     pass


class IndexedValueArray(np.ndarray):
    """
    """
    def __new__(cls, input_array, main_header=None, subheaders=None):
        """
        """
        obj = np.asarray(input_array).view(cls)
        obj.str_len = None
        obj.main_header = main_header
        obj.subheaders = subheaders

        return obj


    def __array_finalize__(self, obj):
        """
        """
        if obj is None: return

        self.str_len = getattr(obj, 'str_len', None)
        self.main_header = getattr(obj, 'main_header', None)
        self.subheaders = getattr(obj, 'subheaders', None)


    def __str__(self):
        """
        """
        if self.ndim == 1:
            arr = self[np.newaxis, :]

        elif self.ndim == 2:
            arr = self

        else:
            return super(IndexedValueArray, self).__str__()

        vsep_1col = '-' * 37 + '\n'
        vsep_2col = '-' * 75 + '\n'

        if arr.main_header:
            if arr.shape[0] == 1:
                s = vsep_1col
                s += '{0:^35}\n'.format(arr.main_header)
            else:
                s = vsep_2col
                s += '{0:^75}\n'.format(arr.main_header)
        else:
            s = ''

        m = arr.shape[0]

        if self.str_len:
            n = min(arr.shape[1], self.str_len)
        else:
            n = arr.shape[1]

        for i in xrange(0, m - m % 2, 2):
            if arr.subheaders:
                s += vsep_2col
                s += ('{0:<25}{1:<15}{2:<25}{3}\n'
                      .format(arr.subheaders[i][0], 
                              arr.subheaders[i][1],
                              arr.subheaders[i+1][0], 
                              arr.subheaders[i+1][1]))
                                      
            s += vsep_2col

            for j in xrange(n):
                a0 = format_entry(arr[i][j][0])
                a1 = format_entry(arr[i][j][1])
                b0 = format_entry(arr[i+1][j][0])
                b1 = format_entry(arr[i+1][j][1])

                s += '{0:<25}{1:<15}{2:<25}{3}\n'.format(a0, a1, b0, b1)

        if m % 2:
            if arr.subheaders:
                s += vsep_1col
                s += ('{0:<25}{1}\n'
                      .format(arr.subheaders[m-1][0], 
                              arr.subheaders[m-1][1]))
                                      
            s += vsep_1col

            for j in xrange(n):
                a0 = format_entry(arr[m-1][j][0])
                a1 = format_entry(arr[m-1][j][1])
                s += '{0:<25}{1}\n'.format(a0, a1)
            
        return s


# TODO: Investigate compressed forms of symmetric matrix. Cf.
# scipy.spatial.distance.squareform
class IndexedSymmArray(np.ndarray):
    """
    """
    def __new__(cls, input_array, labels=None):
        """
        """
        obj = np.asarray(input_array).view(cls)
        obj.labels = labels

        return obj


    def __array_finalize__(self, obj):
        """
        """
        if obj is None: return
        self.labels = getattr(obj, 'labels', None)


############################################################
#                        Testing
############################################################

def test_IndexedValueArray():

    terms = ['row', 'row', 'row', 'your', 'boat', 'gently', 'down', 'the', 
             'stream', 'merrily', 'merrily', 'merrily', 'merrily', 'life', 
             'is', 'but', 'a', 'dream']

    values = [np.random.random() for t in terms]

    d = [('i', np.array(terms).dtype), 
         ('value', np.array(values).dtype)]
    v = np.array(zip(terms, values), dtype=d)

    arr = np.vstack([v] * 5)
    arr = arr.view(IndexedValueArray)
    arr.main_header = 'Test 2-d Array'
    arr.subheaders = [('Repetition ' + str(i), 'Random') 
                      for i in xrange(arr.shape[0])]

    print arr
    print

    arr = v.view(IndexedValueArray)
    arr.main_header = 'Test 1-d Array'

    print arr
