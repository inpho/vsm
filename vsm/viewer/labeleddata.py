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


def format_(x, n):
    """
    """
    if isint(x):
        return x

    if isfloat(x):
        return '{0:.5f}'.format(x)

    n = min(n, len(x))
    return x[:n]


def default_col_widths(dtype):
    """
    """
    col_widths = []
    
    values =zip(*dtype.fields.values())[0]

    for t in values:
        if t.kind == 'S':
            col_widths.append(t.itemsize + 5)
        else:
            col_widths.append(15)
    
    return col_widths


class LabeledColumn(np.ndarray):
    """
    A subclass of nd.ndarray whose purpose is to store labels and
    formatting information for a 1-dimensional structured array. It
    also provides pretty-printing routines.

    A column can have a header and a default display length.

    A subcolumn wraps the data found under a given field name. Each
    subcolumn has a label and a display width.
    """
    def __new__(cls, input_array, col_header=None, subcol_headers=[], subcol_widths=[]):
        """
        """
        obj = np.asarray(input_array).view(cls)
        obj.col_header = col_header
        obj.col_len = None
        obj.subcol_headers = subcol_headers
        if len(subcol_widths) == 0:
            obj.subcol_widths = default_col_widths(input_array.dtype)
        else:
            obj.subcol_widths = subcol_widths
        
        return obj


    def __array_finalize__(self, obj):
        """
        """
        if obj is None: return

        self.col_header = getattr(obj, 'col_header', None)
        self.col_len = getattr(obj, 'col_len', None)
        self.subcol_headers = getattr(obj, 'subcol_headers', None)
        self.subcol_widths = getattr(obj, 'subcol_widths', None)


    def __str__(self):
        
        if self.col_len:
            col_len = min(self.shape[0], self.col_len)
        else:
            col_len = self.shape[0]

        col_width = sum(self.subcol_widths)

        line = '-' * col_width + '\n'
        out = line
        if self.col_header:
            out += '{0:^{1}}'.format(format_(self.col_header, col_width), 
                                     col_width) + '\n'
            out += line
            
        if len(self.subcol_headers) > 0:
            for i in xrange(len(self.subcol_headers)):
                w = self.subcol_widths[i]
                out += '{0:<{1}}'.format(format_(self.subcol_headers[i], w), w)
            out += '\n'
            out += line

        for i in xrange(col_len):
            for j in xrange(len(self.dtype)):
                w = self.subcol_widths[j]
                n = self.dtype.names[j]
                out += '{0:<{1}}'.format(format_(self[n][i], w), w)
            out += '\n'

        return out



class DataTable(list):
    """
    A subclass of list whose purpose is to store labels and
    formatting information for a list of 1-dimensional structured
    arrays. It also provides pretty-printing routines.

    Globally, the table has a default display length for the columns
    and a table header.

    A column can have a column-specific header.

    A subcolumn wraps the data found under a given field name. Each
    subcolumn has a label and a display width.
    """
    def __init__(self, l, table_header=None):
        """
        """
        super(DataTable, self).__init__(l)
        self.table_header = table_header

    def __str__(self):
        """
        """
        col_width = sum(self[0].subcol_widths)


        out = '-' * col_width + '\n'
        if self.table_header:
            out += '{0:^{1}}'.format(format_(self.table_header, col_width), 
                                     col_width) + '\n'

        for col in self:
            out += col.__str__()

        return out



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


def test_LabeledColumn():

    terms = ['row', 'row', 'row', 'your', 'boat', 'gently', 'down', 'the', 
             'stream', 'merrily', 'merrily', 'merrily', 'merrily', 'life', 
             'is', 'but', 'a', 'dream']
    values = [np.random.random() for t in terms]
    d = [('i', np.array(terms).dtype), 
         ('value', np.array(values).dtype)]
    v = np.array(zip(terms, values), dtype=d)
    arr = LabeledColumn(v)
    # arr.subcol_widths = [30, 20]
    arr.subcol_headers = ['Word', 'Value']
    arr.col_headers = 'Song'
    arr.col_len = 10

    return arr


def test_DataTable():

    terms = ['row', 'row', 'row', 'your', 'boat', 'gently', 'down', 'the', 
             'stream', 'merrily', 'merrily', 'merrily', 'merrily', 'life', 
             'is', 'but', 'a', 'dream']
    values = [np.random.random() for t in terms]
    d = [('i', np.array(terms).dtype), 
         ('value', np.array(values).dtype)]
    v = np.array(zip(terms, values), dtype=d)
    v = LabeledColumn(v)
    # v.subcol_widths = [30, 20]
    v.subcol_headers = ['Word', 'Value']
    v.col_len = 10
    t = []
    for i in xrange(5):
        t.append(v.copy())
        t[i].col_header = 'Iteration ' + str(i)
    t = DataTable(t, 'Song')

    return t
