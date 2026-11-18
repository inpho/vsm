from setuptools import setup, Extension
import numpy

# All project metadata and package discovery are configured in pyproject.toml.
# This shim only declares the Cython extension module; because Cython is listed
# in [build-system].requires, setuptools compiles the .pyx source directly.
setup(
    ext_modules=[
        Extension(
            "vsm.model._cgs_update",
            sources=["vsm/model/_cgs_update.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ],
)
