from setuptools import setup, Extension, Command, find_packages
import numpy

from Cython.Build import cythonize

extensions = [Extension(sources=['vsm/model/_cgs_update.pyx'], language='c++',
                include_dirs=[numpy.get_include()], name='vsm.model._cgs_update')]
        #    extra_compile_args=['-march=native'],
        #    extra_link_args=['-march=native'],
        #    define_macros=[('CYTHON_TRACE','1')]

setup(
    pbr=True,
    setup_requires=['pbr'],
    ext_modules = cythonize(extensions),
    package_data = {'vsm': ['vsm/model/_cgs_update.pyx']},
    #dependency_links=['https://inpho.cogs.indiana.edu/pypi/czipfile/',
    #    'https://inpho.cogs.indiana.edu/pypi/pymmseg/'],
    test_suite = "unittest2.collector",
    tests_require=['unittest2'],
)
