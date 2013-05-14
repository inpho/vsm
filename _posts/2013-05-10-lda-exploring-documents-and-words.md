--- 
title: LDA - Exploring documents and words
layout: post
---

##LDA Tutorial: Exploring documents and words

The previous tutorial (link here) illustrated various methods to examine topics in an estimated LDA model. 
In this document we focus on the analysis of documents using LDA.
As before, we use the 100-topics LDA model trained with the [Stanford Encyclopedia of Philosophy](http://plato.stanford.edu/), and assume the viewer object 'v' is created from an appropriate corpus and a model. 

### Exploring documents

LDA provides powerful methods to search, sort and relate documents in the corpus. 

As a first step, we illustrate how to find documents that are related to a particular topic or topics. 
Suppose we are interested in articles in the whole SEP that deals with the classical physics. 
To find them we use `sim_top_doc`. 
As we have seen before (LINK HERE) in our model the classical physics is featured in topic 2. 
So we look for documents related to topic 2:

{% highlight bash %}
$ v.sim_top_doc([2])
{% endhighlight %}

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="2">Topics: 2</th></tr><tr><th style="text-align: center; background: #EFF2FB; ">Document </th><th style="text-align: center; background: #EFF2FB; ">Prob </th></tr><tr><td>newton-principia.txt </td><td>0.92416 </td></tr><tr><td>newton-stm.txt </td><td>0.91340 </td></tr><tr><td>descartes-physics.txt </td><td>0.89946 </td></tr><tr><td>leibniz-physics.txt </td><td>0.87209 </td></tr><tr><td>atomism-modern.txt </td><td>0.85466 </td></tr><tr><td>newton-philosophy.txt </td><td>0.84752 </td></tr><tr><td>galileo.txt </td><td>0.82841 </td></tr><tr><td>spacetime-theories.txt </td><td>0.81533 </td></tr><tr><td>copernicus.txt </td><td>0.76159 </td></tr><tr><td>gassendi.txt </td><td>0.70505 </td></tr></table>

The table lists the top 10 relevant documents, with the corresponding probabilities as a measure of relevance. 

One can also use a set of topics as a query. 
For example, suppose we are interested not only in the classical physics but in the physics in general. 
For this purpose we may first run `sim_top_top([2])` and then use the result (i.e. a set of topics similar to topic 2) as a query for `sim_top_doc`.
Here, we use the top 6 topics most related to topic 2 as a query:

{% highlight bash %}
$ query, prob = zip(*v.sim_top_top([2])[:5])
$ query
(2, 89, 79, 93, 21)
$ v.sim_top_doc(query)
{% endhighlight %}

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="2">Topics: 2, 89, 79, 93, 21</th></tr><tr><th style="text-align: center; background: #EFF2FB; ">Document </th><th style="text-align: center; background: #EFF2FB; ">Prob </th></tr><tr><td>spacetime-iframes.txt </td><td>0.58719 </td></tr><tr><td>time-machine.txt </td><td>0.54281   </td></tr><tr><td>spacetime-theories.txt </td><td>0.52951 </td></tr><tr><td>arabic-islamic-natural.txt </td><td>0.50608 </td></tr><tr><td>spacetime-bebecome.txt </td><td>0.49068 </td></tr><tr><td>equivME.txt </td><td>0.48692 </td></tr><tr><td>time-thermo.txt </td><td>0.48065 </td></tr><tr><td>causation-backwards.txt </td><td>0.47840 </td></tr><tr><td>spacetime-convensimul.txt </td><td>0.47770 </td></tr><tr><td>spacetime-singularities.txt </td><td>0.46816 </td></tr></table>

Note that this time the list contains documents on more general issues.

In LDA, each document assigns probabilities over topics, and thus can be located in the K-dimensional real space, where K is the total number of topics in the model. 
Similarity among topics can thus be defined in this space. 
Currently our similarity function uses cosine values so that two documents with a high cosine value is judged as `similar`. 

To look for documents similar to 'descartes.txt'. Use:

{% highlight bash %}
$ v.sim_doc_doc('descartes.txt')
{% endhighlight %}

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="2">Documents: </th></tr><tr><th style="text-align: center; background: #EFF2FB; ">Document </th><th style="text-align: center; background: #EFF2FB; ">Cosine </th></tr><tr><td>descartes.txt </td><td>1.00000   </td></tr><tr><td>desgabets.txt </td><td>0.93482 </td></tr><tr><td>henricus-regius.txt </td><td>0.90705 </td></tr><tr><td>legrand.txt </td><td>0.90498 </td></tr><tr><td>margaret-cavendish.txt </td><td>0.84320 </td></tr><tr><td>leibniz.txt </td><td>0.83812   </td></tr><tr><td>malebranche.txt </td><td>0.83744 </td></tr><tr><td>cordemoy.txt </td><td>0.82133 </td></tr><tr><td>john-norris.txt </td><td>0.79888 </td></tr><tr><td>spinoza-physics.txt                         </td><td>0.79842   </td></tr></table>

As with topics, one can obtain pairwise similarities for a set of documents in the form of similarity matrix. 
As an example, the similarity matrix for the above five documents can be obtained by:

{% highlight bash %}
$ docs, prob = zip(*v.sim_doc_doc('descartes.txt')[:5])
$ docs
('descartes.txt',
 'desgabets.txt',
 'henricus-regius.txt',
 'legrand.txt',
 'margaret-cavendish.txt')
$ v.simmat_docs(docs)
IndexedSymmArray([[ 1.        ,  0.93481742,  0.90705438,  0.90498041,  0.84319661],
                  [ 0.93481742,  1.        ,  0.8757895 ,  0.91618604,  0.8773178 ],
                  [ 0.90705438,  0.8757895 ,  1.        ,  0.92957827,  0.74286053],
                  [ 0.90498041,  0.91618604,  0.92957827,  1.        ,  0.69783225],
                  [ 0.84319661,  0.8773178 ,  0.74286053,  0.69783225,  1.        ]])
{% endhighlight %}

### Exploring words

A word query is the most common way to search documents. 
In LDA, each occurrence of a word is assigned with its topic value, giving the idea as to in what context the word is used. 
Our word search function thus outputs not only tells us documents that contain the query word, but also its position in the documents and the assigned topic values.
Take for example the term 'anthropomorphism':

{% highlight bash %}
$ v.word_topics('anthropomorphism')
{% endhighlight %}

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="3">Word: anthropomorphism</th></tr><tr><th style="text-align: center; background: #EFF2FB; ">Document </th><th style="text-align: center; background: #EFF2FB; ">Pos </th><th style="text-align: center; background: #EFF2FB; ">Topic </th></tr><tr><td>abraham-daud.txt </td><td>2161 </td><td>19 </td></tr><tr><td>arnauld.txt </td><td>4076 </td><td>19 </td></tr><tr><td>causation-mani.txt </td><td>5906 </td><td>91 </td></tr><tr><td>cognition-animal.txt </td><td>1373 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2006 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2014 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2016 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2035      </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2060 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2086 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2121 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2275 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2354 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>2441 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>3061 </td><td>76 </td></tr><tr><td>cognition-animal.txt </td><td>3770 </td><td>76 </td></tr><tr><td>comte.txt </td><td>2506 </td><td>19 </td></tr><tr><td>consciousness-animal.txt </td><td>1803 </td><td>76 </td></tr><tr><td>consciousness-animal.txt </td><td>1816 </td><td>76 </td></tr><tr><td>ethics-environmental.txt </td><td>3435 </td><td>19 </td></tr><tr><td>feminist-religion.txt </td><td>3171 </td><td>19        </td></tr><tr><td>hume-religion.txt </td><td>4133 </td><td>19 </td></tr><tr><td>hume-religion.txt </td><td>7637 </td><td>19 </td></tr><tr><td>kant-religion.txt </td><td>4596      </td><td>19 </td></tr><tr><td>kukai.txt </td><td>1684      </td><td>31 </td></tr><tr><td>ludwig-feuerbach.txt </td><td>3874      </td><td>19        </td></tr><tr><td>maimonides.txt </td><td>4194 </td><td>19        </td></tr><tr><td>nothingness.txt </td><td>4360 </td><td>76        </td></tr><tr><td>philolaus.txt </td><td>5881 </td><td>31 </td></tr><tr><td>reduction-biology.txt </td><td>149 </td><td>76 </td></tr><tr><td>relativism.txt </td><td>11889     </td><td>76 </td></tr><tr><td>xenophanes.txt </td><td>21 </td><td>19 </td></tr></table>

Hence this word instantiates three topics, 19, 76, 91. These topics are:

{% highlight bash %}
$ v.topics(k_indices=[19, 76, 91])
{% endhighlight %}

<table style="margin: 0"><tr><th style="text-align: center; background: #CEE3F6" colspan ="11">Topics Sorted by Index</th></tr><tr><th style="text-align: center; background: #EFF2FB;" >Topic</th><th style="text-align: center; background: #EFF2FB;" >Words</th></tr><tr><td style="padding-left:0.75em;">19</td><td> god, divine, world, human, religion, theological, power, christian, creation, nature </td></tr><tr><td style="padding-left:0.75em;">76</td><td> behavior, psychology, cognitive, mental, human, mind, psychological, attention, imagery, animals </td></tr><tr><td style="padding-left:0.75em;">91</td><td> one, two, system, set, case, first, given, way, also, example </td></tr></table>

