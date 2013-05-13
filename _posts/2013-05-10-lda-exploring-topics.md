--- 
title: LDA - Exploring topics
layout: post
---

## LDA: Exploring topics

In this page we illustrate various ways to analyze a trained LDA model. As an example, we use a 100-topics LDA model trained with 1553 articles from the [Stanford Encyclopedia of Philosophy](http://plato.stanford.edu/).

First, we load both a corpus (`sep_corpus.npz`) and a trained LDA model (`sep_model.npz`):

```
$ from vsm.util.corpustools import Corpus
$ c = Corpus.load('sep_corpus.npz')
Loading corpus from descartes_corpus.npz
$ from vsm.model.ldagibbs import LDAGibbs as LDA
$ m = LDA.load('sep_model.npz')
Loading LDA-Gibbs data from descartes.npz
```

To analyze an LDA model, we create a viewer object from the corpus and the model:

```
$ from vsm.viewer.ldagibbsviewer import LDAGibbsViewer as LDAViewer
$ v = LDAViewer(c, m)
```

First, let's plot the log probabilities (not available on remote ipython session).  

```
$ v.logp_plot()
<module 'matplotlib.pyplot' from '/Library/Python/2.7/site-packages/matplotlib-1.3.x-py2.7-macosx-10.8-intel.egg/matplotlib/pyplot.pyc'>
```

**TODO: put image here**

The chain seems to be roughly converged. 

### View topics

To see all topics in the model from command line, type:

```
$ print v.topics()
```

This gives a list of all topics, each of which is a list of words and corresponding probabilities. 
In an ip notebook, 

```
$ v.topics()
```

gives a compact html table where each topic is represented by the top 10 probability words:

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="11">Topics Sorted by Index</th></tr><tr><th style="text-align: center; background: #EFF2FB;" >Topic</th><th style="text-align: center; background: #EFF2FB;" >Words</th></tr><tr><td style="padding-left:0.75em;">0</td><td> medieval, abelard, william, socrates, john, ockham, de, bacon, boethius, century </td></tr><tr><td style="padding-left:0.75em;">1</td><td> one, see, example, may, way, might, many, different, kind, things </td></tr><tr><td style="padding-left:0.75em;">2</td><td> motion, newton, space, bodies, force, body, matter, atoms, leibniz, forces </td></tr><tr><td style="padding-left:0.75em;">3</td><td> truth, true, propositions, theory, russell, proposition, facts, false, correspondence, fact </td></tr><tr><td style="padding-left:0.75em;">4</td><td> hegel, berlin, philosophy, german, religion, herder, fichte, jacobi, cohen, history </td></tr><tr><td style="padding-left:0.75em;">5</td><td> x, y, f, 0, set, n, algebra, b, c, function </td></tr><tr><td style="padding-left:0.75em;">...</td><td> ........ </td></tr></table>

If not only words characterizing topics but also there probabilities are needed, type:

```
$ v.topics(compact_view=False)
```

which gives words and their corresponding probability for each topic as:

<table><tr><th style="text-align: center; background: #A9D0F5; fontsize: 14px;" colspan="6"> Topics Sorted by Index </th></tr><tr><th style="text-align: center; background: #CEE3F6;" colspan="2">Topic 0</th><th style="text-align: center; background: #CEE3F6;" colspan="2">Topic 1</th><th style="text-align: center; background: #CEE3F6;" colspan="2">Topic 2</th></tr><tr><th style="text-align: center; background: #EFF2FB;"> Word</th><th style="text-align: center; background: #EFF2FB;"> Prob</th><th style="text-align: center; background: #EFF2FB;"> Word</th><th style="text-align: center; background: #EFF2FB;"> Prob</th><th style="text-align: center; background: #EFF2FB;"> Word</th><th style="text-align: center; background: #EFF2FB;"> Prob</th></tr><tr><td>medieval</td><td>0.01971</td><td>one</td><td>0.04451</td><td>motion</td><td>0.04242</td></tr><tr><td>abelard</td><td>0.01298</td><td>see</td><td>0.01712</td><td>newton</td><td>0.02785</td></tr><tr><td>william</td><td>0.01152</td><td>example</td><td>0.01708</td><td>space</td><td>0.01944</td></tr><tr><td>socrates</td><td>0.01084</td><td>may</td><td>0.01640</td><td>bodies</td><td>0.01890</td></tr><tr><td>john</td><td>0.01016</td><td>way</td><td>0.01537</td><td>force</td><td>0.01337</td></tr><tr><td>ockham</td><td>0.00981</td><td>might</td><td>0.01476</td><td>body</td><td>0.01236</td></tr><tr><td>de</td><td>0.00954</td><td>many</td><td>0.01405</td><td>matter</td><td>0.01198</td></tr><tr><td>bacon</td><td>0.00952</td><td>different</td><td>0.01374</td><td>atoms</td><td>0.01132</td></tr><tr><td>boethius</td><td>0.00908</td><td>kind</td><td>0.01362</td><td>leibniz</td><td>0.00939</td></tr><tr><td>century</td><td>0.00892</td><td>things</td><td>0.01315</td><td>forces</td><td>0.00937</td></tr></table>
...

To see specific topics, use `k_indices` as:

```
$ print v.topics(k_indices=[2,6,13])
```

which lists just three topics, 2, 6 and 13.

### Find similar topics

Looking at the topics shown above topic 2 seems to be related to the classical physics. Are there other topics similar to it? To see similarities between topics, we use `sim_top_top`:

```
$ print v.sim_top_top([2])
....................
     Topics: 2
....................
Topic     Cosine
....................
2         1.00000
89        0.20470
79        0.20083
93        0.19487
21        0.17824
83        0.16438
75        0.11835
36        0.10747
51        0.09974
73        0.09409
```

These are topics similar to topic 2. 
Let's see the top 6 topics from this list using `k_indices` in `topics` method:

```
$ v.topics(k_indices=[2, 89, 79, 93, 21, 83])
```

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="11">Topics Sorted by Index</th></tr><tr><th style="text-align: center; background: #EFF2FB;" >Topic</th><th style="text-align: center; background: #EFF2FB;" >Words</th></tr><tr><td style="padding-left:0.75em;">2</td><td> motion, newton, space, bodies, force, body, matter, atoms, leibniz, forces </td></tr><tr><td style="padding-left:0.75em;">89</td><td> soul, knowledge, human, body, natural, nature, matter, things, mind, material </td></tr><tr><td style="padding-left:0.75em;">79</td><td> time, change, infinite, past, temporal, sequence, state, finite, chance, future </td></tr><tr><td style="padding-left:0.75em;">93</td><td> spacetime, theory, relativity, einstein, field, physical, physics, quantum, general, space </td></tr><tr><td style="padding-left:0.75em;">21</td><td> energy, bohr, principle, quantum, entropy, state, mechanics, correspondence, boltzmann, theory </td></tr><tr><td style="padding-left:0.75em;">83</td><td> descartes, god, leibniz, spinoza, substance, ideas, mind, malebranche, nature, substances </td></tr></table>

Hence the topics related to the general/contemporary physics and the modern philosophy are judged as 'similar' to topic 2.

We can also look at the similarities between each pair from a given set of topics by using `simmat_topics`. 
This will return a numpy array containing the similarity matrix for a given topics 

```
$ v.simmat_topics(k_indices=[2, 89, 79, 93, 21, 83])
IndexedSymmArray([[ 1.        ,  0.20469593,  0.2008326 ,  0.19487031,  0.17824172, 0.16437827],
                  [ 0.20469593,  1.        ,  0.05569496,  0.08155678,  0.07375815, 0.17739257],
                  [ 0.2008326 ,  0.05569496,  1.        ,  0.15310955,  0.19200546, 0.05160133],
                  [ 0.19487031,  0.08155678,  0.15310955,  1.        ,  0.30703874, 0.05445239],
                  [ 0.17824172,  0.07375815,  0.19200546,  0.30703874,  1.        , 0.04247789],
                  [ 0.16437827,  0.17739257,  0.05160133,  0.05445239,  0.04247789, 1.        ]])
```

### Explore topics by document

In LDA, each document in the corpus is assigned with a probability distribution over topics, which characterizes the content of the document. Suppose we are interested in the SEP article on [Descartes](http://plato.stanford.edu/entries/descartes/), and ask which topics are discussed in it. For this we use `doc_topics`:

```
$ print v.doc_topics('descartes.txt')
.......................
Document: descartes.txt
.......................
Topic     Prob
.......................
83        0.21227
89        0.19220
82        0.10778
9         0.08628
2         0.08313
59        0.04773
70        0.04472
51        0.04415
48        0.04071
52        0.03239
```

Let's look at the top five topics:

```
$ v.topics(k_indices=[83, 89, 82, 9, 2])
```

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="11">Topics Sorted by Index</th></tr><tr><th style="text-align: center; background: #EFF2FB;" >Topic</th><th style="text-align: center; background: #EFF2FB;" >Words</th></tr><tr><td style="padding-left:0.75em;">83</td><td> descartes, god, leibniz, spinoza, substance, ideas, mind, malebranche, nature, substances </td></tr><tr><td style="padding-left:0.75em;">89</td><td> soul, knowledge, human, body, natural, nature, matter, things, mind, material </td></tr><tr><td style="padding-left:0.75em;">82</td><td> work, published, first, time, new, one, years, also, could, book </td></tr><tr><td style="padding-left:0.75em;">9</td><td> would, even, whether, could, two, however, since, rather, another, also </td></tr><tr><td style="padding-left:0.75em;">2</td><td> motion, newton, space, bodies, force, body, matter, atoms, leibniz, forces </td></tr></table>

In this list we recognize the terms related to the modern philosophy (topic 83), the mind & body problem (topic 89), and the classical physics (topic 2), as is expected.

### Explore topics by words

One can also ask which topics are most relevant to a given word. This gives the contexts in which a particular word is used in the corpus. Let's take the word "anthropomorphism" for example and see which topics are related to this word.
To this we use `sim_word_top`: 

```
$ v.sim_word_top('anthropomorphism')
```

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="11">Sorted by Word Similarity</th></tr><tr><th style="text-align: center; background: #EFF2FB;" >Topic</th><th style="text-align: center; background: #EFF2FB;" >Words</th></tr><tr><td style="padding-left:0.75em;">76</td><td> behavior, psychology, cognitive, mental, human, mind, psychological, attention, imagery, animals </td></tr><tr><td style="padding-left:0.75em;">19</td><td> god, divine, world, human, religion, theological, power, christian, creation, nature </td></tr><tr><td style="padding-left:0.75em;">31</td><td> world, one, reality, within, experience, process, human, time, self, individual </td></tr><tr><td style="padding-left:0.75em;">...</td><td> .........</td></tr></table>

Topic 76 looks phycology, whereas topic 19 is about theology. 
So we see that "anthropomorphism" is discussed at least in these two contexts. 
This makes sense, for in phycology it is often discussed whether one can legitimately project human abilities to animals, whereas in theology the anthropomorphism has been a traditional contention in the god-human relationship.

### Cluster topics

When there are lot of topics, one may have a group of related topics. In such cases, it is useful to see clusters of topics. Our LDA viewer supports k-means, spectral clustering and affinity propagation as clustering algorithms. For a description of each algorithm see e.g. [here](http://scikit-learn.org/stable/modules/clustering.html).

Here we use k-means to illustrate topic clusters in our LDA model. 
k-means algorithm requires cluster number to be fixed. We choose 10 clusters.

```
$ cls = v.cluster_topics(method='k-means', n_clusters=10)
Initialization complete
Iteration 0, inertia 149.883810088
Iteration 1, inertia 84.901591324
Iteration 2, inertia 84.7033890556
Iteration 3, inertia 84.4245218161
Converged to similar centers at iteration 3
```

`cls` now contains a list of 10 clusters:

```
$ cls
[[19, 23, 31, 69, 83],
 [4, 7, 10, 13, 18, 27, 34, 35, 50, 53, 57, 59, 61, 86, 98],
 [1, 6, 9, 40, 47, 48, 52, 70, 82, 88, 91],
 [2, 21, 28, 51, 72, 75, 79, 93],
 [15, 24, 30, 45, 58, 65, 81, 85, 87],
 [5, 12, 42, 46, 55, 64, 84, 90],
 [11, 16, 17, 20, 29, 33, 38, 41, 43, 44, 56, 63, 66, 67, 74, 77, 78, 96],
 [8, 32, 36, 49, 73, 76, 80, 92, 97, 99],
 [0, 14, 25, 26, 39, 62, 89, 94, 95],
 [3, 22, 37, 54, 60, 68, 71]]
```

One can look at each cluster by using `topics` function:

```
$ v.topics(k_indices=cls[0])
```

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="11">Topics Sorted by Index</th></tr><tr><th style="text-align: center; background: #EFF2FB;" >Topic</th><th style="text-align: center; background: #EFF2FB;" >Words</th></tr><tr><td style="padding-left:0.75em;">19</td><td> god, divine, world, human, religion, theological, power, christian, creation, nature </td></tr><tr><td style="padding-left:0.75em;">23</td><td> possible, worlds, world, modal, true, object, w, actual, objects, kripke </td></tr><tr><td style="padding-left:0.75em;">31</td><td> world, one, reality, within, experience, process, human, time, self, individual </td></tr><tr><td style="padding-left:0.75em;">69</td><td> god, theism, hartshorne, evil, world, universe, existence, chisholm, whitehead, process </td></tr><tr><td style="padding-left:0.75em;">83</td><td> descartes, god, leibniz, spinoza, substance, ideas, mind, malebranche, nature, substances </td></tr></table>

Which looks like a mixture of theological / modern philosophy and possible world semantics.
As another example, let's look at cluster 5: 

```
$ v.topics(k_indices=cls[5])
```

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="11">Topics Sorted by Index</th></tr><tr><th style="text-align: center; background: #EFF2FB;" >Topic</th><th style="text-align: center; background: #EFF2FB;" >Words</th></tr><tr><td style="padding-left:0.75em;">5</td><td> x, y, f, 0, set, n, algebra, b, c, function </td></tr><tr><td style="padding-left:0.75em;">12</td><td> x, set, theory, sets, y, frege, type, ph, f, axiom </td></tr><tr><td style="padding-left:0.75em;">42</td><td> mathematics, mathematical, proof, godel, numbers, hilbert, logic, brouwer, intuitionistic, arithmetic </td></tr><tr><td style="padding-left:0.75em;">46</td><td> logic, logical, reasoning, formal, calculus, rules, inference, default, ai, form </td></tr><tr><td style="padding-left:0.75em;">55</td><td> p, 1, b, 2, 3, see, 4, two, following, section </td></tr><tr><td style="padding-left:0.75em;">64</td><td> probability, evidence, e, h, probabilities, hypothesis, hypotheses, inductive, induction, p </td></tr><tr><td style="padding-left:0.75em;">84</td><td> logic, ph, m, b, l, g, formula, formulas, logics, semantics </td></tr><tr><td style="padding-left:0.75em;">90</td><td> b, p, belief, lewis, conditional, set, conditionals, k, w, probability </td></tr></table>

which forms a more coherent cluster relating to logics / formal epistemology.

Note: the clustering algorithms used in our viewer are stochastic, so you should expect to get different clusterings each time you execute the function.
