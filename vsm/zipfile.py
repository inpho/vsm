"""
Monkey-patch for numpy to use czipfile rather than zipfile
"""

import imp
import sys

import zipfile
print "replacing zipfile with czipfile"
moduleName = 'zipfile'
tmpModuleName = 'czipfile'

if imp.lock_held() is True:
    del sys.modules[moduleName]
    sys.modules[tmpModuleName] = __import__(tmpModuleName)
    sys.modules[moduleName] = __import__(tmpModuleName)

print "zipfile optimized"
