#!/bin/bash
CMD="coverage run -a --source=vsm"
CMD="coverage run -a --source vsm.model,vsm.viewer,vsm.corpus,vsm.spatial,vsm.split,vsm.structarr,vsm.exceptions"
#CMD="coverage run -a --source vsm.model,vsm.viewer,vsm.corpus,vsm.spatial,vsm.split,vsm.structarr,vsm.exceptions --debug trace"

rm -rf .coverage
coverage debug sys

$CMD -m pytest unit_tests/*
EXIT=$?

rm -rf ap.ini ap ap.tgz
#pip install --pre topicexplorer
#$CMD -m topicexplorer.demo
#EXIT=$?+$EXIT
#$CMD -m topicexplorer.serve ap.ini 

coverage report

echo "Test exit code: $EXIT"
exit $EXIT
