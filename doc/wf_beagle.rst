============
Beagle Model
============

Here are the following variations of Beagle Model.

The Beagle model was first featured in `Representing Word Meaning and Order Information in a Composite Holographic Lexicon <http://www.indiana.edu/~clcl/New/Jones_Mewhort_PR.pdf>`_ by Jones and Mewhort.

:mod:`Beagle Composite<vsm.model.beaglecomposite>` - the Beagle Composite model 
combines Beagle Context and Beagle Order model in user-defined ratio.

Beagle Context model has 2 convenient classes for training.

:mod:`Beagle Context<vsm.model.beaglecontext>` - To get a better sense of a word, Beagle context model
considers the co-occurring words.

Beagle Order model has 2 convenient classes for training.

:mod:`Beagle Order<vsm.model.beagleorder>` - considers the order of a word in the sentence.

:mod:`Bealge Environment<vsm.model.beagleenvironment>` - random environment vector model.

Here is a blog entry about `BEAGLE model<https://inpho.cogs.indiana.edu/datablog/info/info-beagle/>`_ in the InPho DataBlog.

.. .. toctree::
    :maxdepth: 1

..  beaglecomposite
    beaglecontextseq
    beaglecontextmulti
    beagleorderseq
    beagleordermulti
    beagleenvironment

