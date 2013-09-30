=======
Viewers
=======

Here are the viewer classes that correspond to each model.
These viewer objects include model-specific functions to visualize
semantic distance between words, documents, and topics (for LDA).

:mod:`Beagle viewer<vsm.viewer.beagleviewer>` - used with any 
variations of the Beagle model.
             
:mod:`LDA Gibbs viewer<vsm.viewer.ldagibbsviewer>` - used with models
:class:`LDA Gibbs<vsm.model.ldagibbs.LDAGibbs>` or 
:class:`LDA CGS Multi<vsm.model.ldacgsmulti.LdaCgsMulti>`.

:mod:`LSA viewer<vsm.viewer.lsaviewer>` - used with
:mod:`LSA model<vsm.model.lsa>` model.

:mod:`Tf viewer<vsm.viewer.tfviewer>` - used with models
:class:`Tf Seq<vsm.model.tf.TfSeq>` or 
:class:`LDA CGS Multi<vsm.model.tf.TfSeq>`.

:mod:`TfIdf viewer<vsm.viewer.tfidfviewer>` - used with the 
:mod:`Tf-Idf model<vsm.model.tfidf>`.

Now, Take your :class:`vsm.corpus.Corpus` and :mod:`vsm.model` to
explore your corpus using the viewer classes.

.. automodule:: vsm.viewer

.. toctree::
   :maxdepth: 1

   beagleviewer
   ldagibbsviewer
   lsaviewer
   tfviewer
   tfidfviewer
