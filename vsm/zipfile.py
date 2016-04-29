"""
Monkey-patch for numpy to use czipfile rather than zipfile
"""
from __future__ import absolute_import

import imp
import sys

import zipfile


def inject_czipfile():
    _inject('zipfile','czipfile')


def inject_zipfile():
    _inject('zipfile','zipfile')

def _inject(module, new_module):
    if imp.lock_held() is True:
        if sys.modules.get(module):
            del sys.modules[module]
        sys.modules[new_module] = __import__(new_module)
        sys.modules[module] = __import__(new_module)

def use_czipfile(f):
    def wrapper(*args, **kwargs):
        inject_czipfile()
        value = f(*args, **kwargs)
        inject_zipfile()
        return value

    return wrapper

