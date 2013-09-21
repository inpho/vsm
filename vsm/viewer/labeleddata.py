import numpy as np

from vsm import enum_sort, map_strarr, isfloat, isint, isstr


def format_entry(x):
    """
    Formats floats to 5 decimal points and returns a string.
    If `x` is a tuple, all elements in the tuple are formatted.

    :param x: Float to be truncated to 5 decimal points.
    :type x: float or tuple

    :returns: `x` as a string.
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
    Truncates `x` given the length `n`. If `x` is a float 
    it returns floats formatted to 5 decimal points.

    :param x: Element to be truncated.
    :type x: string or integer/float
    
    :param n: Length that we want `x` to be.
    :type n: integer

    :returns: formatted `x` given the length `n`.
    """
    if isint(x):
        return x

    if isfloat(x):
        return '{0:.5f}'.format(x)

    n = min(n, len(x))
    return x[:n]


def default_col_widths(dtype):
    """
    Assigns default column widths depending on the dtype. 
    Used in _str_ representation.
    """
    col_widths = []
    
    values =zip(*dtype.fields.values())[0]

    for t in values:
        if t.kind == 'S':
            col_widths.append(t.itemsize + 5)
        else:
            col_widths.append(10)
    
    return col_widths


def compact_col_widths(dtype, n):
    """
    Assigns second column width CompactList based on the dtype. 
    """
    ccol_widths = [10, 0]

    value =zip(*dtype.fields.values())[0][0]

    for i in xrange(n):
        if value.kind == 'S':
            ccol_widths[1] += value.itemsize + 2
        else:
            ccol_widths[1] += 10
   
    return ccol_widths


class LabeledColumn(np.ndarray):
    """
    A subclass of np.ndarray whose purpose is to store labels and
    formatting information for a 1-dimensional structured array. It
    also provides pretty-printing routines.

    A column can have a header and a default display length.

    A subcolumn wraps the data found under a given field name. Each
    subcolumn has a label and a display width.

    :param input_array: Array to be formatted into a LabeledColumn.
    :type input_array: 1-dimensional structured array

    :param col_header: The title of the object. For example, 'Words: logic'.
        Default is `None`.
    :type col_header: string, optional
    
    :param subcol_headers: List of labels that correspond to the fields of 
        the structured array. Default is `None`.
    :type subcol_headers: list, optional

    :param subcol_widths: List of widths for each subcolumn. If not provided, 
        `subcol_widths` is calculated based on the data-type of the entries.
    :type subcol_widths: list, optional
    
    :param col_len: Number of entries to display. If not provided, `col_len`
        is set to length of LabeledColumn.
    :type col_len: integer, optional

    :attributes:
        * **col_header** (string, optional)
            The title of the object. For example, 'Words: logic'.
            Default is `None`.
        * **subcol_headers** (list, optional)
            List of labels that correspond to the fields of 
            the structured array. Default is `None`.

        * **subcol_widths** (list, optional)
            List of widths for each subcolumn. If not provided, 
            `subcol_widths` is calculated based on the data-type of the entries.
    
        * **col_len** (integer, optional) 
            Number of entries to display. If not provided, `col_len`
            is set to length of LabeledColumn.

    :methods:
        * **__str__**
            Returns a pretty printed string version of the object.
        * **_repr_html_**
            Returns a html table in ipython online session.

    **Examples**
    
    >>>  words = ['there','will','be','an','answer']
    >>>  values = [random.random() for w in words]
    >>>  arr = np.array(zip(words, values), 
            dtype=[('i', np.array(words).dtype), 
            ('value', np.array(values).dtype)])
    >>>  lc = LabeledColumn(arr)
    >>>  lc.col_header = 'Words'
    >>>  lc.subcol_headers = ['Word', 'Value']
    >>>  lc.subcol_widths
    [11, 10]
    >>>  lc.col_len
    5
    >>> print lc
    Words        
    ---------------------
    Word       Value     
    ---------------------
    there      0.22608   
    will       0.64567   
    be         0.02832   
    an         0.31118   
    answer     0.23083   

    """
    def __new__(cls, input_array, col_header=None, subcol_headers=None,
                subcol_widths=None, col_len=None):
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
        """
        Pretty prints the LabeledColumn when 'print' method is used.
        """     
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
        Returns an html table in ipython online session.
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


class CompactTable(np.ndarray):
    """
    A subclass of np.ndarray whose purpose is to store labels
    and formatting information for 2-dimensional structured array.
    It also provides pretty-printing routines.

    :param input_array: Array to be formatted into a CompactTable.
    :type input_array: 2-dimensional structured array

    :param table_header: The title of the object. Default is `None`.
    :type table_header: string, optional
    
    :param subcol_headers: List of labels that correspond to the columns
        of the table. Default is `None`.
    :type subcol_headers: list, optional

    :param first_cols: List of strings that describes the values in the
        second column. If not provided, an item in `first_cols` is 'Topic n'
        where 'n' is the index in the `first_cols`.
    :type first_cols: list, optional
    
    :param subcol_widths: List of widths for each subcolumn. If not provided, 
        `subcol_widths` is calculated based on the data-type of the entries.
    :type subcol_widths: list, optional
    
    :param num_words: Number of entries to display in the second column. 
        If not provided, `num_words` is set to 5.
    :type num_words: integer, optional

    :attributes:
        * **table_header** (string, optional)
            The title of the object.
        * **subcol_headers** (list, optional)
            List of labels that correspond to the fields of 
            the structured array. Default is `None`.
        * **first_cols** (list, optional)
            List of strings that describes the values in the second column.
        * **subcol_widths** (list, optional)
            List of widths for each subcolumn. If not provided, 
            `subcol_widths` is calculated based on the data-type of the entries.
        * **num_words** (integer, optional) 
            Number of entries to display.

    :methods:
        * **__str__**
            Returns a pretty printed string version of the object.
        * **_repr_html_**
            Returns a html table in ipython online session.

    **Examples**
    
    >>> li = [[('logic', 0.08902691511387155), ('foundations', 0.08902691511387155),
            ('computer', 0.08902691511387155), ('theoretical', 0.059449866903282994)], 
         [('calculus', 0.14554670528602476), ('lambda', 0.14554670528602476),
          ('variety', 0.0731354091238234), ('computation', 0.0731354091238234)],
         [('theology', 0.11285794497473327), ('roman', 0.11285794497473327),
          ('catholic', 0.11285794497473327), ('source', 0.05670971364402021)]]
    >>> arr = np.array(li, dtype=[('words', '|S16'), ('values', '<f8')])
    >>> ct = CompactTable(arr, table_header='Compact view', subcol_headers=['Topic', 'Words'],
                        num_words=4)
    >>> print ct
    --------------------------------------------------
                       Compact view                   
    --------------------------------------------------
    Topics    Words                                   
    --------------------------------------------------
     Topic 0 logic     foundationcomputer  theoretica
     Topic 1 calculus  lambda    variety   computatio
     Topic 2 theology  roman     catholic  source    
    """
    def __new__(cls, input_array, table_header=None, subcol_headers=None,
                first_cols=None, subcol_widths=None, num_words=None):
        """
        """
        obj = np.asarray(input_array).view(cls)
        obj.table_header = table_header
        obj.subcol_headers = subcol_headers
        obj._first_cols = first_cols
        obj._subcol_widths = subcol_widths     
        obj._num_words = num_words        
        return obj


    def __array_finalize__(self, obj):
        """
        """
        if obj is None: return

        self.table_header = getattr(obj, 'table_header', None)
        self.subcol_headers = getattr(obj, 'subcol_headers', None)
        self._first_cols = getattr(obj, '_first_cols', None)
        self._subcol_widths = getattr(obj, '_subcol_widths', None)
        self._num_words = getattr(obj, '_num_words', None)

    @property
    def subcol_widths(self):
        if not hasattr(self, '_subcol_widths') or not self._subcol_widths:
            self._subcol_widths = compact_col_widths(self.dtype, self.num_words)
        return self._subcol_widths

    @subcol_widths.setter
    def subcol_widths(self, w):
        self._subcol_widths = w

    @property
    def first_cols(self):
        if not hasattr(self, '_first_cols') or not self._first_cols:
            self._first_cols = ['Topic ' + str(i) for i in xrange(len(self))]
        return self._first_cols

    @first_cols.setter
    def first_cols(self, w):
        self._first_cols = w

    @property
    def num_words(self):
        if not self._num_words:
            return 5
        return self._num_words

    @num_words.setter
    def num_words(self, n):
        self._num_words = n


    def __str__(self):
        """
        Pretty prints `CompatTable` when `print` method is used.
        """     
        width = sum(self.subcol_widths)
        line = '-' * width + '\n'
        out = line
        if self.table_header:
            out += '{0:^{1}}'.format(format_(self.table_header, width), 
                                     width) + '\n'
            out += line
            
        if self.subcol_headers:
            for i in xrange(len(self.subcol_headers)):
                w = self.subcol_widths[i]
                out += '{0:<{1}}'.format(format_(self.subcol_headers[i], w), w)
            out += '\n'
            out += line

        for i in xrange(len(self)):
            w = self.subcol_widths[0] - 2
            out += '  {0:<{1}}'.format(format_(self.first_cols[i], w), w)

            for j in xrange(self.num_words):
                n = self.dtype.names[0] 
                w = self.subcol_widths[1] / self.num_words
                out += '{0:<{1}}'.format(format_(self[i][n][j], w), w)
            out += '\n'

        return out


    def _repr_html_(self):
        """
        Returns an html table in ipython online session.
        """ 
        s = '<table style="margin: 0">'

        if self.table_header:
            s += '<tr><th style="text-align: center; background: #CEE3F6" colspan\
            ="{0}">{1}</th></tr>'.format(1 + self.num_words, self.table_header)

        if self.subcol_headers:
            s += '<tr>'
            for i, sch in enumerate(self.subcol_headers):
                s += '<th style="text-align: center; background: #EFF2FB;" \
                    >{0}</th>'.format(sch)
            s += '</tr>'
        
        for i in xrange(len(self)):
            s += '<tr>'
            w = self.subcol_widths[0]
            s += '<td style="padding-left:0.75em;">{0}</td>'.format(
                    format_(self.first_cols[i], w), w)

            s += '<td>' 
            for j in xrange(self.num_words):
                n = self.dtype.names[0] 
                w = self.subcol_widths[1] / self.num_words
                if j == self.num_words -1:
                    s += ' {0:<{1}}'.format(self[i][n][j], w)
                else:
                    s += ' {0},'.format(self[i][n][j])
            s += '</td>'
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
 
    :param l: List of 1-dimensional structured arrays.
    :type l: list
    
    :param table_header: The title of the object. Default is `None`.
    :type table_header: string, optional
   
    :attributes:
        * **table_header** (string)
            The title of the object. Default is `None`.
  
    :methods:
        * **__str__**
            Returns a pretty printed string version of the object.
        * **_repr_html_**
            Returns an html table in ipython online session.

    **Examples**
    
    >>>  words = ['there','will','be','an','answer']
    >>>  values = [random.random() for w in words]
    >>>  arr = np.array(zip(words, values), 
            dtype=[('i', np.array(words).dtype), 
            ('value', np.array(values).dtype)])
    >>>  lc = LabeledColumn(arr)
    >>>  l = [lc.copy() for i in xrange(2)]
    >>>  dt = DataTable(l, 'Let it be')
    >>>  dt.table_header
    Let it be
    >>>  print dt
        Let it be      
    ---------------------
            Words        
    ---------------------
    Word       Value     
    ---------------------
    there      0.58793   
    will       0.29624   
    be         0.00209   
    an         0.27221   
    answer     0.96118   
    ---------------------
            Words        
    ---------------------
    Word       Value     
    ---------------------
    there      0.22608   
    will       0.64567   
    be         0.02832   
    an         0.31118   
    answer     0.23083      
    """
    
    def __init__(self, l, table_header=None):
        super(DataTable, self).__init__(l)
        self.table_header = table_header
        
    def __getslice__(self, i, j):
        """
        """
        return DataTable(list.__getslice__(self, i, j), 
                    table_header=self.table_header)
                    
    def __str__(self):
        """
        Pretty prints the DataTable when `print` method is used.
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
        Returns a html table in ipython online session.
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
                        w = lc.subcol_widths[j]
                        n = lc.dtype.names[j]
                        s += '<td>{0}</td>'.format(format_(lc[n][i], w))
    
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
