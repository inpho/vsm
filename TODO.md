VSM HTRC Extensions
---------------------
Witht his branch, I am to restructure the vsm.extensions.htrc module to include
complete downloading of an HTRC collection, migration to using either raw JSON
or a CouchDB store for metadata, and having generic closures for htrc label
functions.

The current implementation revolves around specific functions per htrc corpus.
These can be generalized into two corpus types: by volume or by page. In order
to deal with the new Darwin corpus, I will work on the by volume functions
first.
