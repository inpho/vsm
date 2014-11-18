This directory contains python and java code that interacts with
java code retrieved from LdaGibbsSamlper.java at http://knowceans.com.


Directions
----------
0. write corpus txtfile:   
            from FileReadWrite import write_file
			write_file(Corpus, ctx_type, 'fname.txt')
    
1. compile:  javac -cp py4j0.8.1.jar *.java    (in gibbstest dir)

2. run:      java -cp org/knowceans/gibbstest/py4j0.8.1.jar:.
                      org.knowceans.gibbstest.LDA
		      org/knowceans/gibbstest/testcorp.txt
	         (in parent dir of org)

3. python:   run LdaRoutine  (in ipython)

4. exit out of java program to end the server connection.

Notes
-----
- directory structure: org/knowceans/gibbstest

- running java starts the gateway server. This needs to be running for python 
code (py4j) to work.

- needs java version "1.7.0_25" to run correctly.

- all java files are in package org.knowceans.gibbstest;

- LDA.java takes written corpus file (from 0.) as args

- python code works with LdaGibbsSampler java object.

- LdaRoutine.py depends on vsm, so move LdaRoutine.py and FileReadWrite.py to a location where vsm is importable, if needed.
