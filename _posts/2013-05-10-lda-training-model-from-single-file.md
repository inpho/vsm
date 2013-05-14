--- 
title: LDA - Training a model from a single file
layout: post
---

##LDA: Training a model from a single file

This page illustrates one simple example of LDA analysis using a text downloadable from the Project Gutenberg. As an example, we choose Descartes' [selections from the Principles of Philosophy](http://www.gutenberg.org/ebooks/4391). Download the plain text file and place it in a working directory with an appropriate name (here we call it `descartes.txt`). If necessary, you can cut off the proviso and save the main text only.
 
First we create a corpus file using `file_corpus`. Type following lines in ipython session (or in ip notebook):

{% highlight bash %}
In [1]: from vsm.util.corpustools import file_corpus
In [2]: c = file_corpus('descartes.txt')
Computing collection frequencies
Selecting words of frequency <= 1
Removing stop words
Rebuilding corpus
Recomputing token breaks: paragraph
Recomputing token breaks: sentence
{% endhighlight %}

Now the corpus is created, as we can see by:

{% highlight bash %}
In [3]: c
Out[3]: <vsm.corpus.Corpus at 0x10867bcd0>
{% endhighlight %}

A corpus making tool tokenizes the text at several levels, which are referred to as `context_type`. Available context type(s) can be checked by: 

{% highlight bash %}
In [4]: c.context_types
Out[4]: ('paragraph', 'sentence')
{% endhighlight %}

Next, we create an LDA model with this corpus. Import an LDA as:

{% highlight bash %}
In [5]: from vsm.model.ldagibbs import LDAGibbs as LDA
{% endhighlight %}

In creating an LDA model, we must specify (1) context type that LDA regard as one document and (2) the number of topics in the model. Here we choose 'paragraph' as context and fix the topic number to 10:

{% highlight bash %}
In [6]: m = LDA(c, context_type='paragraph', K=10)
{% endhighlight %}

Now we are ready to run an LDA analysis on the text. Let's try 30 iterations first. This should take several minutes depending on environment:

{% highlight bash %}
In [7]: m.train(itr=30)
iteration 29
{% endhighlight %}

We can continue to sample further to get a better convergence:

{% highlight bash %}
In [8]: m.train(itr=70)
iteration 99
{% endhighlight %}

The model can be saved for a later analysis or training:

{% highlight bash %}
In [9]: m.save('descartes')
Saving LDA-Gibbs model to descartes
{% endhighlight %}

This creates a file `descartes.npz`. To load the saved model:

{% highlight bash %}
In [10]: m2 = LDA.load('descartes.npz')
Loading LDA-Gibbs data from descartes.npz
{% endhighlight %}

Analyzing an LDA model also requires the corresponding corpus. So it is recommended to save the corpus as well:

{% highlight bash %}
In [11]: c.save('descartes_corpus')
Saving corpus as descartes_corpus
{% endhighlight %}

To load a corpus, we use Corpus object:

{% highlight bash %}
In [12]: from vsm.util.corpustools import Corpus
In [13]: c2 = Corpus.load('descartes_corpus.npz')
Loading corpus from descartes_corpus.npz
{% endhighlight %}
