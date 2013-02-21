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
	if not self._col_len:
	    return self.shape[0]
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
        s = '<table style="margin: 0">'

        if self.col_header:
            s += '<tr><th style="text-align: center; background: #CEE3F6" colspan\
		="{0}">{1}</th></tr>'.format(len(self.subcol_widths), self.col_header)

        if self.subcol_headers:
            s += '<tr>'
            for sch in self.subcol_headers:
                s += '<th style="text-align: center; background: #EFF2FB; ">{0}\
			</th>'.format(sch)
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
	s = '<table>'

	col_in_row = 3

        if self.table_header:
            s += '<tr><th style="text-align: center; background: #A9D0F5;\
		fontsize: 14px;" colspan="{0}"> {1} </th></tr>'.format(col_in_row
			 * len(self[0].subcol_headers), self.table_header)
     
	start = 0
	n_arr = len(self)
	m = n_arr % col_in_row

        while start < n_arr:
	    end = start + col_in_row
	    group = self[start:end]	    
	    
	    s += '<tr>'
	    for i, lc in enumerate(group):
	
		if lc.col_header:
	            s += '<th style="text-align: center; background: #CEE3F6;"\
		 colspan="{0}">{1}</th>'.format(len(lc.subcol_headers), lc.col_header)

		if end > n_arr and m and i == len(group)-1 and start > 0:
		    for j in xrange(end - n_arr):
			s += '<th style="border-color: #EFF2FB; background: #EFF2FB;"\
		 	colspan="{0}"> {1}</th>'.format(len(lc.subcol_headers), 
					' ' * self[0].col_width)
    	    s += '</tr>'

	    s += '<tr>'
	    for i, lc in enumerate(group):
        	
		if lc.subcol_headers:
            	    
            	    for sch in lc.subcol_headers:
                	s += '<th style="text-align: center; background: #EFF2FB;">\
				{0}</th>'.format(sch)

		if end > n_arr and m and i == len(group)-1 and start > 0:
		    for j in xrange(end - n_arr):
			s += '<th style="border-color: #EFF2FB; background: #EFF2FB;"\
		 colspan="{0}"> {1}</th>'.format(len(lc.subcol_headers), 
						' ' * self[0].col_width)
            s += '</tr>'
            
	    for i in xrange(self[0].col_len):

	        s += '<tr>'
		for k, lc in enumerate(group):
            	   
            	    for j in xrange(len(lc.dtype)):
                        n = lc.dtype.names[j]
	                s += '<td>{0}</td>'.format(lc[n][i])
		
		    if end > n_arr and m and k == len(group)-1 and start > 0:
		        for e in xrange(end - n_arr):
			    s += '<td style="border-color: #EFF2FB; background: #EFF2FB;"\
				colspan="{0}"> {1} </th>'.format(len(lc.subcol_headers),
							 ' ' * self[0].col_width)
                s += '</tr>'	
	    
	    start = end
        s += '</table>'

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

def test_LabeledColumn():

    words = ['row', 'row', 'row', 'your', 'boat', 'gently', 'down', 'the', 
             'stream', 'merrily', 'merrily', 'merrily', 'merrily', 'life', 
             'is', 'but', 'a', 'dream']
    values = [np.random.random() for t in words]
    d = [('i', np.array(words).dtype), 
         ('value', np.array(values).dtype)]
    v = np.array(zip(words, values), dtype=d)
    arr = v.view(LabeledColumn)
#    arr.subcol_widths = [30, 20]
    arr.subcol_headers = ['Word', 'Value']
    arr.col_header = 'Song lets make this longer than subcol headers'
    arr.col_len = 10

    return arr


def test_DataTable():

    words = ['row', 'row', 'row', 'your', 'boat', 'gently', 'down', 'the', 
             'stream', 'merrily', 'merrily', 'merrily', 'merrily', 'life', 
             'is', 'but', 'a', 'dream']
    values = [np.random.random() for t in words]
    d = [('i', np.array(words).dtype), 
         ('value', np.array(values).dtype)]
    v = np.array(zip(words, values), dtype=d)
    v = LabeledColumn(v)
    v.subcol_widths = [30, 20]
    v.subcol_headers = ['Word', 'Value']
    v.col_len = 10
    t = []
    for i in xrange(5):
        t.append(v.copy())
        t[i].col_header = 'Iteration ' + str(i)
    t = DataTable(t, 'Song')

    return t
