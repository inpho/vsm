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
            col_widths.append(10)
    
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
    def __new__(cls, input_array, col_header=None, subcol_headers=None,
                subcol_widths=None, col_len = None):
        """
        """
        obj = np.asarray(input_array).view(cls)
        obj.col_header = col_header
        obj._col_len = col_len
        obj.subcol_headers = subcol_headers
        obj._subcol_widths = subcol_widths     
        
        return obj


    def __array_finalize__(self, obj):
        """
        """
        if obj is None: return

        self.col_header = getattr(obj, 'col_header', None)
        self._col_len = getattr(obj, '_col_len', None)
        self.subcol_headers = getattr(obj, 'subcol_headers', None)
        self._subcol_widths = getattr(obj, '_subcol_widths', None)


    @property
    def subcol_widths(self):
        if not hasattr(self, '_subcol_widths') or not self._subcol_widths:
            self._subcol_widths = default_col_widths(self.dtype)
        return self._subcol_widths

    @subcol_widths.setter
    def subcol_widths(self, w):
        self._subcol_widths = w

    @property
    def col_width(self):
        return max(sum(self.subcol_widths), len(self.col_header))

    @property
    def col_len(self):
	return min(self.shape[0], self._col_len)

    @col_len.setter
    def col_len(self, n):
	self._col_len = n


    def __str__(self):
         
        line = '-' * self.col_width + '\n'
        out = line
        if self.col_header:
            out += '{0:^{1}}'.format(format_(self.col_header, self.col_width), 
                                     self.col_width) + '\n'
            out += line
            
        if self.subcol_headers:
            for i in xrange(len(self.subcol_headers)):
                w = self.subcol_widths[i]
                out += '{0:<{1}}'.format(format_(self.subcol_headers[i], w), w)
            out += '\n'
            out += line

        for i in xrange(self.col_len):
            for j in xrange(len(self.dtype)):
                w = self.subcol_widths[j]
                n = self.dtype.names[j]
                out += '{0:<{1}}'.format(format_(self[n][i], w), w)
            out += '\n'

        return out


    def _repr_html_(self):
        """
	""" 
	s = '<style> th {text-align: center; background: #E0F8F7; } </style>'
        s += '<table style="margin: 0">'

        if self.col_header:
            s += '<tr><th style="text-align: center;" colspan="{0}"> {1}\
		 </th></tr>'.format(len(self.subcol_widths), self.col_header)

        if self.subcol_headers:
            s += '<tr>'
            for sch in self.subcol_headers:
                s += '<th style="text-align: center;">{0}</th>'.format(sch)
            s += '</tr>'
        
        for i in xrange(self.col_len):
            s += '<tr>'
            for j in xrange(len(self.dtype)):
                w = self.subcol_widths[j]
                n = self.dtype.names[j]
                s += '<td>{0:<{1}}</td>'.format(format_(self[n][i], w), w)
            s += '</tr>'
        
        s += '</table>'
 
        return s



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
        col_width = self[0].col_width

        out = '-' * col_width + '\n'
        if self.table_header:
            out += '{0:^{1}}'.format(format_(self.table_header, col_width), 
                                     col_width) + '\n'

        for col in self:
            out += col.__str__()

        return out

    

    def _repr_html_(self):
	"""
	"""        
	s = '<style> th {text-align: center; background: #E0F8F7; } </style>'
	s += '<table>'

	col_in_row = 3

        if self.table_header:
            s += '<tr><th style="text-align: center;" colspan="{0}"> {1} </th>\
		</tr>'.format(col_in_row * len(self[0].subcol_headers), self.table_header)
     
	start = 0

        while start < len(self):
	    end = start + col_in_row

	    s += '<tr>'
	    for lc in self[start:end]:
	        
		if lc.col_header:
	            s += '<th style="text-align: center;" colspan="{0}"> {1}\
			</th>'.format(len(lc.subcol_headers), lc.col_header)
	    s += '</tr>'

	    s += '<tr>'
	    for lc in self[start:end]:
        	
		if lc.subcol_headers:
            	    
            	    for sch in lc.subcol_headers:
                	s += '<th style="text-align: center;">{0}</th>'.format(sch)
            s += '</tr>'
            
	    for i in xrange(self[0].col_len):
	        s += '<tr>'
		for lc in self[start:end]:
            	   
            	    for j in xrange(len(lc.dtype)):
                        n = lc.dtype.names[j]
	                s += '<td>{}</td>'.format(lc[n][i])
                s += '</tr>'	
	    
	    start = end

        s += '</table>'

        return s



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
    arr = v.view(LabeledColumn)
#    arr.subcol_widths = [30, 20]
    arr.subcol_headers = ['Word', 'Value']
    arr.col_header = 'Song lets make this longer than subcol headers'
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
    v.subcol_widths = [30, 20]
    v.subcol_headers = ['Word', 'Value']
    v.col_len = 10
    t = []
    for i in xrange(7):
        t.append(v.copy())
        t[i].col_header = 'Iteration ' + str(i)
    t = DataTable(t, 'Song')

    return t
