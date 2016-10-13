#!/bin/bash


n=`ps aux | grep translate1.py | wc -l`
if [ $n -lt 2 ]
then
  echo $n
  ./run.sh 1>>info1 2>>info2

fi
