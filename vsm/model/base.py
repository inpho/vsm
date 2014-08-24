import numpy as np


__all__ = ['BaseModel']




class BaseModel(object):
    """
    Base class for models which store data in a single matrix.

    :param matrix: A two-dimensional numpy array storing the results
        of model training. Default is `None`.
    :type matrix: numpy.ndarray, optional
    
    :param context_type: A string specifying the type of context over
        which the model trainer is applied. Default is `None`.
    :type context_type: string, optional

    :attributes:
        Same as parameters.

    :methods:
        * **save**
            Takes a filename or file object and saves `self.matrix` 
            in an npz archive.
        * **load**
            Takes a filename or file object and loads it as an npz
            archive into a BaseModel object.

    :See Also: :meth:`numpy.savez`, :meth:`numpy.load`
    """
    def __init__(self, matrix=None, context_type=None):
        self.matrix = matrix
        self.context_type = context_type


    def save(self, f):
        """
        Takes a filename or file object and saves `self.matrix` in an
        npz archive.
        
        :param file: Designates the file to which to save data. See
            `numpy.savez` for further details.
        :type file: str-like or file-like object
            
        :returns: `None`

        :See Also: :meth:`numpy.savez`
        """
        print 'Saving model to', f
        np.savez(f, matrix=self.matrix, context_type=np.array(self.context_type))


    @staticmethod
    def load(f):
        """
        Takes a filename or file object and loads it as an npz archive
        into a BaseModel object.

        :param file: Designates the file to read. If `file` is a string
            ending in `.gz`, the file is first gunzipped. See `numpy.load`
            for further details.
        :type file: str-like or file-like object

        :returns: A dictionary storing the data found in `file`.

        :See Also: :meth:`numpy.load`
        """
        print 'Loading model from', f
        npz = np.load(f)
        
        # The slice [()] is to unwrap sparse matrices, which get saved
        # in singleton object arrays
        return BaseModel(matrix=npz['matrix'][()], context_type=npz['context_type'][()])
