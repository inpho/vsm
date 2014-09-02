import warnings
from functools import update_wrapper, wraps
import inspect

__all__ = ['deprecation_warning', 'deprecated_meth']



def deprecation_warning(old_name, new_name):
    """
    Deprecation warning for deprecated functions.
    """
    warnings.simplefilter('always', DeprecationWarning)
    
    message = "{0} is deprecated. Please use {1} instead.".format(old_name,
                new_name)
    warnings.warn(message, DeprecationWarning)


#TODO: a function for deprecated class AND auto generate doc string with
# a note about deprecation.

def deprecated_meth(new_fn_name):
    """
    Decorator to be used for deprecated functions/modules.
    Throws a DeprecationWarning.
    """
    def wrap(old_fn):
        
        def wrapper(self, *args, **kwargs):
            new_fn = getattr(self, new_fn_name)
            deprecation_warning(old_fn.__name__, new_fn.__name__)
            
            return new_fn(*args, **kwargs)
  
        #update_wrapper(wrapper, new_fn_)
        return wrapper

    return wrap


"""
def deprecated_fn(new_fn):
    Decorator to be used for deprecated functions/modules.
    Throws a DeprecationWarning.
    def wrap(old_fn):
        
        def wrapper(self, *args, **kwargs):
            deprecation_warning(old_fn.__name__, new_fn.__name__)
            #new_fn_ = getattr(self, new_fn.__name__)

            return new_fn(self, *args, **kwargs)
  
        update_wrapper(wrapper, new_fn)
        
        return wrapper

    return wrap
"""
