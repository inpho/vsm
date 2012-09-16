from distutils.core import setup

import setuptools

setup(
    name = "vsm",
    version = "0.1",
    description = ('Vector Space Semantic Modeling Framework '\
                   'for the Indiana Philosophy Ontology Project'),
    author = "The Indiana Philosophy Ontology (InPhO) Project",
    author_email = "inpho@indiana.edu",
    url = "http://inpho.cogs.indiana.edu/",
    download_url = "http://www.github.com/inpho/vsm",
    keywords = [],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # "License :: OSI Approved :: MIT License", TBD
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Text Processing :: Linguistic",
        ],
    install_requires=[
        "numpy>=1.6.1",
        "scipy>=10.1",
        "nltk>=2.0.0"
    ],
    packages=['vsm',
              'vsm.corpus',
              'vsm.model',
              'vsm.viewer'],
)
