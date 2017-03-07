#!/bin/bash

FOLDER=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/dev
PROGRAM=/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v4
TESTFOLDER=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/official-test

RESULTFOLDER=results
MODEL=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/scripts-9-run-v4/models/prefix=pos30chunk30

PREFIX=pos30chunk30-

python $PROGRAM/evaluate.py --test_folder $TESTFOLDER --out_folder $RESULTFOLDER --model $MODEL --prefix $PREFIX
