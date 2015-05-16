from __future__ import unicode_literals

import numpy as np

from vsm.structarr import *
from types import *


__all__ = ['DataTable', 'IndexedSymmArray', 'LabeledColumn']



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


def default_col_widths(dtype, col_header):
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
            col_widths.append(len(col_header) + 5)
    
    return col_widths


def calc_col_num(col_len, n):
    
    num = col_len / n

    if col_len % n > 0:
        num += 1

    return num


def max_col_num(li, max_width):
    """
    Calculates the total number of columns for multi_col option
    in LabeledColumn.
    """
    w = sum(li)
    num = max_width/w
    if num == 0:
        return 1
    return num


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

    :param multi_col: If `True` html table version of :class:`LabeledColumn`
        is displayed in multiple columns when the number of entries exceed 15.
        Default is `True`.
    :type multi_col: boolean, optional
        
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
             subcol_widths=None, col_len=None, col_num=None, multi_col=True):
        """
        """
        obj = np.asarray(input_array).view(cls)
        obj.col_header = col_header
        obj._col_len = col_len
        obj.subcol_headers = subcol_headers
        obj._subcol_widths = subcol_widths     
        obj._col_num = col_num
        obj.multi_col = multi_col

        return obj


    def __array_finalize__(self, obj):
        """
        """
        if obj is None: return

        self.col_header = getattr(obj, 'col_header', None)
        self._col_len = getattr(obj, '_col_len', None)
        self.subcol_headers = getattr(obj, 'subcol_headers', None)
        self._subcol_widths = getattr(obj, '_subcol_widths', None)
        self._col_num = getattr(obj, '_col_num', None)
        self.multi_col = getattr(obj, 'multi_col', True)

    @property
    def col_num(self):
        if not hasattr(self, '_col_num') or not self._col_num:
            self._col_num = min(calc_col_num(self._col_len, 15),
                                max_col_num(self._subcol_widths, 160))
        return self._col_num

    @col_num.setter
    def col_num(self, w):
        self._col_num = w

    @property
    def subcol_widths(self):
        if not hasattr(self, '_subcol_widths') or not self._subcol_widths:
            self._subcol_widths = default_col_widths(self.dtype, self.col_header)
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
        if self.multi_col and self.col_len >15:
            if self.col_header:
                s += '<tr><th style="text-align: center; background: #CEE3F6" colspan\
                    ="{0}">{1}</th></tr>'.format(len(self.subcol_widths)*self.col_num, 
                    self.col_header)
            
            if self.subcol_headers:
                s += '<tr>'
                subcol = ''
                for sch in self.subcol_headers:
                    subcol += '<th style="text-align: center; background: #EFF2FB; ">{0}\
                    </th>'.format(sch)
                s += subcol * self.col_num
                s += '</tr>'
            
            count = self.col_len
            last_row = self.col_len % self.col_num
            rows = self.col_len / self.col_num
            
            li = [rows] * self.col_num
            li = [li[i]+1 if i<last_row else li[i] for i in xrange(self.col_num)]
            li = [0] + li[:-1]
            if last_row > 0:
                rows += 1            

            for k in xrange(rows):
                s += '<tr>'
                ind = k
                for i in xrange(self.col_num):
                    ind += li[i]
                    for j in xrange(len(self.dtype)):
                        w = self.subcol_widths[j]
                        n = self.dtype.names[j]
                        if count > 0:
                            s += '<td>{0:<{1}}</td>'.format(format_(self[n][ind],w), w)
                        else:
                            s += '<td style="border-color: #EFF2FB; background: #EEF2FB;">\
                             {0}</td>'.format(' '* w)
                    count -= 1    
                s += '</tr>'
        else:
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
 
    :param l: List of 1-dimensional structured arrays.
    :type l: list
    
    :param table_header: The title of the object. Default is `None`.
    :type table_header: string, optional
  
    :param compact_view: If `True` the DataTable is displayed with its
        tokens only without the probabilities. Default is `True`
    :type compact_view: boolean, optional

    :attributes:
        * **table_header** (string)
            The title of the object. Default is `None`.
        * **compact_view** (boolean)
            Option of viewing tokens with or without the probabilities.
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
    >>>  lc = LabeledColumn(arr, 'Lyrics')
    >>>  l = [lc.copy() for i in xrange(2)]
    >>>  dt = DataTable(l, 'Let it be', subcolhdr_compact=['Topic', 'Words'],
                   subcolhdr_full=['Word', 'Prob'], compact_view=True)
    >>>  print dt
    --------------------------------------------
                     Let it be                  
    --------------------------------------------
    Topic      Words      
    --------------------------------------------
    Lyrics     there      will       be         
               an         answer     
    --------------------------------------------
    Lyrics     there      will       be         
               an         answer     
    --------------------------------------------

    >>> dt.compact_view = False
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
    
    def __init__(self, l, table_header=None, compact_view=True,
        subcolhdr_compact=None, subcolhdr_full=None):
        """
        """
        super(DataTable, self).__init__(l)
        self.table_header = table_header
        self.compact_view = compact_view
        self.subcolhdr_compact = subcolhdr_compact
        self.subcolhdr_full = subcolhdr_full

    
    def __getslice__(self, i, j):
        """
        """
        return DataTable(list.__getslice__(self, i, j), 
                    table_header=self.table_header, 
                    compact_view=self.compact_view,
                    subcolhdr_compact=self.subcolhdr_compact,
                    subcolhdr_full=self.subcolhdr_full)
                
    def __str__(self):
        """
        Pretty prints the DataTable when `print` method is used.
        """
        if self.compact_view:
            return self.__str_compact__(self.subcolhdr_compact)
        else:
            return self.__str_full__(self.subcolhdr_full)


    def __str_compact__(self, subcol_headers):
        """
        Prints DataTable when `compact_view` is `True`.
        """
        num_cols = 3
        w1 = self[0].subcol_widths[1]
        w2 = min(self[0].subcol_widths[0], 23)
        col_width = w1 + w2 * 3  
        
        line = '-' * col_width + '\n'
        out = line
        if self.table_header:
            out += '{0:^{1}}'.format(format_(self.table_header, col_width), 
                                     col_width) + '\n'
            out += line
        
        if subcol_headers:
            out += '{0:<{1}}'.format(format_(subcol_headers[0], w1), w1)  
            out += '{0:<{1}}\n'.format(format_(subcol_headers[1], w2), w2)
            out += line
        
        for i in xrange(len(self)):
            j = 0
            # topic info
            out += '{0:<{1}}'.format(format_(self[i].col_header, w1), w1)
            # words or tokens
            for idx in xrange(self[i].col_len):
                if j == 0 and idx > 0 :
                    out += " " * w1
                word = self[i].dtype.names[0]
                out += '{0:<{1}}'.format(format_(self[i][word][idx], w2-1), w2)
                j += 1
                if j == num_cols or idx == self[i].col_len -1:
                    out += '\n'
                    j = 0
            out += line
        
        return out
    

    def __str_full__(self, subcol_headers):
        """
        Prints DataTable when `compact_view` is `False`.
        """
        col_width = self[0].col_width

        out = '-' * col_width + '\n'
        if self.table_header:
            out += '{0:^{1}}'.format(format_(self.table_header, col_width), 
                                     col_width) + '\n'

        for col in self:
            col.subcol_headers = subcol_headers
            out += col.__str__()

        return out


    def _repr_html_(self):
        """
        Returns a html table in ipython online session. 
        """
        if self.compact_view:    
            return self._repr_html_compact_(self.subcolhdr_compact)
        else:
            return self._repr_html_full_(self.subcolhdr_full)


    def _repr_html_compact_(self, subcol_headers):
        """
        Returns a html table in ipython online session when 
        `compact_view` is `True`.
        """
        s = '<table style="margin: 0">'

        num_cols = 3
        if self.table_header:
            s += '<tr><th style="text-align: center; background: #CEE3F6" colspan\
            ="{0}">{1}</th></tr>'.format(1 + self[0].col_len, self.table_header)
           
        s += '<tr>'
        for i, sch in enumerate(subcol_headers):
            s += '<th style="text-align: center; background: #EFF2FB;" \
                 >{0}</th>'.format(sch)
        s += '</tr>'
        
        for i in xrange(len(self)):
            s += '<tr>'
            w = self[0].subcol_widths[1]
            s += '<td style="padding-left:0.75em;">{0}</td>'.format(
                    format_(self[i].col_header, w), w)
# line break.
            s += '<td>' 
            for j in xrange(self[0].col_len):
                n = self[0].dtype.names[0] 
                w = self[0].subcol_widths[0]
                if j == self[0].col_len -1:
                    s += ' {0:<{1}}'.format(self[i][n][j], w)
                else:
                    s += ' {0},'.format(self[i][n][j])
            s += '</td>'
            s += '</tr>'
        s += '</table>'
 
        return s

    
    def _repr_html_full_(self, subcol_headers):
        """
        Returns a html table in ipython online session when
        `compact_view` is `False`.
        """        
        s = '<table>'

        col_in_row = 3
        
        n_cols = len(subcol_headers)
        col_w = self[0].col_width

        if self.table_header:
            s += '<tr><th style="text-align: center; background: #A9D0F5;\
            fontsize: 14px;" colspan="{0}"> {1} </th></tr>'.format(col_in_row
             * n_cols, self.table_header)
     
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
                 colspan="{0}">{1}</th>'.format(n_cols, lc.col_header)

                if end > n_arr and m and i == len(group)-1 and start > 0:
                    for j in xrange(end - n_arr):
                        s += '<th style="border-color: #EFF2FB; background: #EFF2FB;"\
                        colspan="{0}"> {1}</th>'.format(n_cols, ' ' * col_w)
            s += '</tr>'

            s += '<tr>'
            for i, lc in enumerate(group):
                    
                for sch in subcol_headers:
                    s += '<th style="text-align: center; background: #EFF2FB;">\
                        {0}</th>'.format(sch)

                if end > n_arr and m and i == len(group)-1 and start > 0:
                    for j in xrange(end - n_arr):
                        s += '<th style="border-color: #EFF2FB; background: #EFF2FB;"\
                         colspan="{0}"> {1}</th>'.format(n_cols, ' ' * col_w)
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
                            colspan="{0}"> {1} </th>'.format(n_cols, ' ' * col_w)
                s += '</tr>'    
            
            start = end
        s += '</table>'

        return s


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
