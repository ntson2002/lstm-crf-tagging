#!/usr/bin/env bash

#
# Train on DELL computer (150.65.242.42)
#
MY_HOME=/home/sonnguyen/jaist

FOLDER=$MY_HOME/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data-japanese/train-dev-test
PROGRAM=$MY_HOME/Bitbucket/lstm-crf-tagging/lstm-tagger-v4

EPOCH=200
TYPE=sgd-lr_.002
TAGSCHEME=iobes
CAPDIM=0
ZERO=0
LOWER=0
WORDDIM=50
WORDLSTMDIM=50
CRF=1
RELOAD=0
CHARDIM=0

#FEATURE=f3.3.15,f5.5.5,f6.6.2
FEATURE=f3.3.15,f6.6.2

ALLEMB=0

###########
FOLD=8
BESTFILE=best.$FOLD.p00c00.txt
python $PROGRAM/train.py --char_dim $CHARDIM --word_lstm_dim $WORDLSTMDIM --train $FOLDER/train.$FOLD.txt --dev $FOLDER/dev.$FOLD.txt --test $FOLDER/test.$FOLD.txt --best_outpath $BESTFILE --reload 0 --lr_method $TYPE --word_dim $WORDDIM --tag_scheme $TAGSCHEME --cap_dim $CAPDIM --zeros $ZERO --lower $LOWER --reload $RELOAD --external_features $FEATURE --epoch $EPOCH --crf $CRF --prefix=$FOLD --all_emb $ALLEMB

