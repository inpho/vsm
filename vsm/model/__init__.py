import numpy as np



class BaseModel(object):
    """
    Base class for models which store data in a single matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        A two-dimensional numpy array storing the results of model
        training. Default is `None`.
    context_type : string
        A string specifying the type of context over which the model
        trainer is applied. Default is `None`.

    Attributes
    ----------
    Same as parameters.

    Methods
    -------
    save
        Takes a filename or file object and saves `self.matrix` in an
        npz archive.
    load
        Takes a filename or file object and loads it as an npz archive
        into a BaseModel object.

    See Also
    --------
    numpy.savez
    numpy.load
    """
    def __init__(self, matrix=None, context_type=None):
        self.matrix = matrix
        self.context_type = context_type


    def save(self, f):
        """
        Takes a filename or file object and saves `self.matrix` in an
        npz archive.
        
        Parameters
        ----------
        file : str-like or file-like object
            Designates the file to which to save data. See
            `numpy.savez` for further details.
            
        Returns
        -------
        None

        See Also
        --------
        BaseModel.load
        numpy.savez
        """
        print 'Saving model to', f
        np.savez(f, matrix=self.matrix, context_type=np.array(self.context_type))


    @staticmethod
    def load(f):
        """
        Takes a filename or file object and loads it as an npz archive
        into a BaseModel object.

        Parameters
        ----------
        file : str-like or file-like object
            Designates the file to read. If `file` is a string ending
            in `.gz`, the file is first gunzipped. See `numpy.load`
            for further details.

        Returns
        -------
        A dictionary storing the data found in `file`.

        See Also
        --------
        BaseModel.save
        numpy.load
        """
        print 'Loading model from', f
        npz = np.load(f)
        
        # The slice [()] is to unwrap sparse matrices, which get saved
        # in singleton object arrays
        return BaseModel(matrix=npz['matrix'][()], context_type=npz['context_type'][()])
