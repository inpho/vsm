mahout.py contains methods that interact with mahout-generated files, 
create vsm `Corpus` and `LDAGibbs`. 


STEPS TO LDA
------------
1) Convert directory of documents to SequenceFile format
mahout-distribution-0.9/bin/mahout seqdirectory -i inpho/testdata -o inpho/mahout-out

2) Creating Vectors from SequenceFile
mahout-distribution-0.9/bin/mahout seq2sparse -i inpho/mahout-out -o inpho/mahout-vect-test

3) Creating Matrix from tf-vectors
mahout-distribution-0.9/bin/mahout rowid -i inpho/mahout-vect-test/tf-vectors -o inpho/mahout-mat-test

4) Run LDA Collapsed Variable Bayes
mahout-distribution-0.9/bin/mahout cvb -i inpho/mahout-mat-test/matrix -dict inpho/mahout-vect-test/dictionary.file-0 -o inpho/mahout-lda-test -a 0.01 -e 0.01 -dt inpho/mahout-dt-test -mt inpho/mahout-models-test -k 5 -x 100


CREATING READABLE FILES
-----------------------
doori@space:~$ mahout-distribution-0.9/bin/mahout vectordump -i inpho/mahout-vect-test/tf-vectors/part-r-00000 -o inpho/mahout-vect-test/tf-vectors/tf.txt -p true --csv csv

doori@space:~$ mahout-distribution-0.9/bin/mahout seqdumper -i inpho/mahout-dt/part-m-00000 -o inpho/mahout-dt/doc-topics.txt

doori@space:~$ mahout-distribution-0.9/bin/mahout seqdumper -i inpho/mahout-lda/part-m-00000 -o inpho/mahout-lda/lda.txt


NOTES
-----
If you are running 'seq2sparse' for building the feature vectors and are using the Lucene  StandardAnalyzer (which is the default), the English stopwords should be removed automatically. (-x option to remove *high frequency* words. default is 99)

REFERENCES
----------
https://mahout.apache.org/users/basics/creating-vectors-from-text.html

http://mahout.apache.org/users/clustering/lda-commandline.html
