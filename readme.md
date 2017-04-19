# Nested named entity recognition using multilayer recurrent networks

* **lstm-tagger-v4**: Implementation of BI-LSTM-CRF with additional features 

* **multi-lstm**: Implementation of Multilayer BI-LSTM-CRF model 

# lstm-tagger-v4 
(Revised: March 28)

## BI-LSTM-CRF with features with 1 layer 

## Multilayer BI-LSTM-CRF

### Initial setup

To use the tagger, you need Python 2.7, with Numpy and Theano installed.

### Input data:
Input files for the training script have to follow the same format than the CoNLL2003 sharing task: each word has to be on a separate line, and there must be an empty line after each sentence. 

Each line contains a words, its features and the ground truth tags at all layers. The below example show the format of our training data file for nested NER. Column 0 is the head words, column 1 and 2 is POS and CHUNK features, column 3 and 4 is groun truth tags at layer 1 and 2. This data sets has 2 type of features and two layers.



```
 col_0      col_1  col_2  col_3  col_4 
---------------------------------------
 Chủ_tịch   N      B-NP   O      O     
 UBND       Ny     B-NP   B-ORG  O     
 tỉnh       N      I-NP   I-ORG  B-LOC 
 Quảng_Nam  NNP    I-NP   I-ORG  I-LOC 
 Nguyễn     NNP    B-NP   B-PER  O     
 Xuân       NNP    I-NP   I-PER  O     
 Phúc       NNP    I-NP   I-PER  O     
 cho        V      B-VP   O      O     
 biết       V      B-VP   O      O     
 tỉnh       N      B-NP   O      O     
 sẽ         R      O      O      O     
 đầu_tư     V      B-VP   O      O     
 xây_dựng   V      B-VP   O      O     
 khu_phố    N      B-NP   O      O     
 đi         V      B-VP   O      O     
 bộ         N      B-NP   O      O     
 tại        E      B-PP   O      O     
 6          M      B-NP   O      O     
 đường      N      B-NP   O      O     
 phố        N      B-NP   O      O     
 chính      A      B-AP   O      O     
 của        E      B-PP   O      O     
 khu        N      B-NP   O      O     
 phố        N      B-NP   O      O     
 cổ         A      B-AP   O      O     
 Hội_An     NNP    B-NP   B-LOC  O     
 ,          CH     O      O      O     
 đồng_thời  C      O      O      O     
 tổ_chức    V      B-VP   O      O     
 chợ        N      B-NP   O      O     
```
### Train a model

To train your own NER model, you need to use the train.py script and provide the location of the training, development and testing set. Below is the script for training our best models. 

* The line `FEATURE=pos.1.30,chunk.2.30` describe the features and feature embedding size. It has the following format: `name1.col_index1.embedding_size1,name2.col_index2.embedding_size2,..`.
* `GOLDCOLUMNS=3,4`: indexes of the ground truth columns in training data file.

```sh 
FOLDER=[Data folder]
PROGRAM=[Program folder]

EPOCH=150
TYPE=sgd-lr_.002
TAGSCHEME=iobes
CAPDIM=3
ZERO=0
LOWER=0
WORDDIM=100
WORDLSTMDIM=100
CRF=1
RELOAD=0
CHARDIM=25

FEATURE=pos.1.30,chunk.2.30  

GOLDCOLUMNS=3,4

ALLEMB=1
PREEMB=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/pre-trained/train.txt.w2vec100

###########
PREFIX=p30c30
BESTFILE=best.$PREFIX.txt
python $PROGRAM/train.py --char_dim $CHARDIM --word_lstm_dim $WORDLSTMDIM --train $FOLDER/train2.conll --dev $FOLDER/dev2.conll --test $FOLDER/testb2.conll --best_outpath $BESTFILE --reload $RELOAD --lr_method $TYPE --word_dim $WORDDIM --tag_scheme $TAGSCHEME --cap_dim $CAPDIM --zeros $ZERO --lower $LOWER --reload $RELOAD --external_features $FEATURE --epoch $EPOCH --crf $CRF --pre_emb $PREEMB --prefix=$PREFIX --tag_columns_string $GOLDCOLUMNS

```

The trained will be saved at a subfolder folder of the folder `models`. The name of this folder is automatically set by the program.

### Use the trained model to tag the test data
