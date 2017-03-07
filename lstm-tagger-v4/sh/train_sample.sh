#!/bin/bash
#PBS -q APPLI
#PBS -N p30c30_iobes
#PBS -j oe

cd $PBS_O_WORKDIR
setenv PATH ${PBS_O_PATH}

# uv
FOLDER=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/dev
PROGRAM=/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v4
PREEMB=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/pre-trained/w2v.100
WORDDIM=100
OUTPUT=best_pos30_chunk30_w2v100_iobes.txt
TAGSCHEME=iobes
PREFIX=pos30chunk30
FEATURE=pos.1.30,chunk.2.30
# Model location: ./models/tag_scheme=iobes,lower=False,zeros=False,char_dim=25,char_lstm_dim=25,char_bidirect=True,word_dim=100,word_lstm_dim=100,word_bidirect=True,pre_emb=w2v.100,all_emb=False,cap_dim=3,pos_dim=30,chunk_dim=30,crf=True,dropout=0.5,lr_method=sgd-lr_.002

python $PROGRAM/train.py --tag_scheme iobes --word_dim $WORDDIM --best_outpath $OUTPUT --pre_emb $PREEMB --train $FOLDER/train.conll  --dev $FOLDER/val.conll --test $FOLDER/test.conll --lower 0 --zeros 0 --char_dim 25 --cap_dim 3 --lr_method sgd-lr_.002 --word_bidirect 1 --external_features $FEATURE --prefix=$PREFIX --reload 0 --epoch 120
