from setuptools import setup, Extension, Command, find_packages
import platform


# find packages in vsm subdirectory
# this will skip the unittests, etc.
packages = ['vsm.'+pkg for pkg in find_packages('vsm')]
packages.append('vsm')

install_requires=[
        "numpy>=1.6.1",
        "scipy>=0.13.0",
        "progressbar>=2.3",
        "chardet>=2.3.0",
        "sortedcontainers>=1.4.0"]

setup(
    name = "vsm",
    version = "0.4.0a29",
    description = ('Vector Space Semantic Modeling Framework '\
                   'for the Indiana Philosophy Ontology Project'),
    author = "The Indiana Philosophy Ontology (InPhO) Project",
    author_email = "inpho@indiana.edu",
    url = "http://inpho.cogs.indiana.edu/",
    download_url = "http://www.github.com/inpho/vsm",
    keywords = [],
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        ],
    install_requires=install_requires,
    license = 'MIT',
    packages=packages,
    ext_modules = [
        Extension('_cgs_update', ['vsm/model/_cgs_update.c']),
    ],
    test_suite = "unit_tests"
)
