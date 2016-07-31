#!/bin/bash
CMD="coverage run -a --source=vsm"
CMD="coverage run -a --source vsm.model,vsm.viewer,vsm.corpus,vsm.spatial,vsm.split,vsm.structarr,vsm.exceptions"
#CMD="coverage run -a --source vsm.model,vsm.viewer,vsm.corpus,vsm.spatial,vsm.split,vsm.structarr,vsm.exceptions --debug trace"

rm -rf .coverage
coverage debug sys

$CMD setup.py test

#pip install --pre topicexplorer
rm -rf ap.ini ap ap.tgz
$CMD -m topicexplorer.demo
#$CMD -m topicexplorer.serve ap.ini 

coverage report
