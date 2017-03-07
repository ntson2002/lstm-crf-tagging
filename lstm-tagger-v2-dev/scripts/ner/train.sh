#!/bin/bash
FOLDER=/Users/$USER/Bitbucket/lstm-crf-tagging/lstm-tagger-v2/data/ner
PROGRAM=/Users/sonnguyen/Bitbucket/lstm-crf-tagging/lstm-tagger-v2
EVALTOOL=/Users/sonnguyen/Bitbucket/lstm-crf-tagging/lstm-tagger-v2/evaluation


#ln -s $EVALTOOL evaluation
python $PROGRAM/train.py --train $FOLDER/train.conll  --dev $FOLDER/val.conll --test $FOLDER/test.conll