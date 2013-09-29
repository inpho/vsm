=======
Viewers
=======

Here are our available viewers that correspond to the models.
These viewer objects uses the trained model to calculate distance between
words, documents, and topics (for LDA).


:mod:`Beagle viewer<vsm.viewer.beagleviewer>` - used with any variations of the Beagle model.

:mod:`LDA Gibbs viewer<vsm.viewer.ldagibbsviewer>` - used with models
:class:`LDA Gibbs<vsm.model.ldagibbs.LDAGibbs>` or 
:class:`LDA CGS Multi<vsm.model.ldacgsmulti.LdaCgsMulti>`.

:mod:`LSA viewer<vsm.viewer.lsaviewer>` - used with any variations of the
:mod:`LSA model<vsm.model.lsa>`.

:mod:`Tf viewer<vsm.viewer.tfviewer>` - used with models
:class:`Tf Seq<vsm.model.tf.TfSeq>` or 
:class:`LDA CGS Multi<vsm.model.tf.TfSeq>`.

:mod:`TfIdf viewer<vsm.viewer.tfidfviewer>` - used with the 
:mod:`Tf-Idf model<vsm.model.tfidf>`.


Now, Take your :class:`vsm.corpus.Corpus` and :mod:`vsm.model` explore your corpus
using the viewer that corresponds to the model.

.. automodule:: vsm.viewer

.. toctree::
   :maxdepth: 1

   beagleviewer
   ldagibbsviewer
   lsaviewer
   tfviewer
   tfidfviewer
