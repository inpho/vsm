"""
The :mod:`vsm` module provides tools and a workflow for producing
semantic models of textual corpora and analyzing and visualizing these
models.

The :mod:`vsm` module has been conceived within the SciPy ecosystem.
In a typical work flow, a collection of texts is first transformed
into a Corpus object, whose underlying data structures are NumPy
numerical arrays. The user may then feed a Corpus object to one of the
model classes, which contain the algorithms, implemented in NumPy,
SciPy and IPython.parallel, for training models such as :doc:`TF<wf_tf>`,
:doc:`TFIDF<wf_tfidf>`, :doc:`LSA<wf_lsa>`,
:doc:`BEAGLE<wf_beagle>`, or :doc:`LDA<wf_lda>`.
Finally, the user may examine the
results with a Viewer class specialized to a particular model type. A
Viewer object contains a variety of methods for analysis and
visualization and achieves its full functionality within an IPython
notebook session extended with matplotlib and scikit-learn.
"""


import vsm.corpus
from vsm.corpus import *
import vsm.model
from vsm.model import *
import vsm.viewer
from vsm.viewer import *

__versi1.0b1'

__all__ = ['__version__']
__all__ += vsm.corpus.__all__[:]
__all__ += vsm.model.__all__
__all__ += vsm.viewer.__all__

