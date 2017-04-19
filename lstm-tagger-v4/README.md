# BI-LSTM-CRF with features with 1 layer (lstm-tagger-v4)



### Train a model
#### Input file

Input files for the training script have to follow the same format than the CoNLL2003 sharing task: each word has to be on a separate line, and there must be an empty line after each sentence. A line must contain at least 2 columns, the first one being the word itself, the last one being the named entity (follow IOB notation). Other columns are features of words.

File `sample_train1.conll`. 

```
 col_0      col_1  col_2  col_3 
--------------------------------
 Chủ_tịch   N      B-NP   O     
 UBND       Ny     B-NP   B-ORG 
 tỉnh       N      I-NP   I-ORG 
 Quảng_Nam  NNP    I-NP   I-ORG 
 Nguyễn     NNP    B-NP   B-PER 
 Xuân       NNP    I-NP   I-PER 
 Phúc       NNP    I-NP   I-PER 
 cho        V      B-VP   O     
 biết       V      B-VP   O     
 tỉnh       N      B-NP   O     
 sẽ         R      O      O     
 đầu_tư     V      B-VP   O     
 xây_dựng   V      B-VP   O     
 khu_phố    N      B-NP   O     
 đi         V      B-VP   O     
 bộ         N      B-NP   O     
 tại        E      B-PP   O     
 6          M      B-NP   O     
 đường      N      B-NP   O     
 phố        N      B-NP   O     
 chính      A      B-AP   O     
 của        E      B-PP   O     
 khu        N      B-NP   O     
 phố        N      B-NP   O     
 cổ         A      B-AP   O     
 Hội_An     NNP    B-NP   B-LOC 
```

#### Train
To train your own model, you need to use the train.py script and provide the location of the training, development and testing set:

Program: `train.py`

Line `FEATURE=pos.1.30,chunk.2.30` described features for training the models. This feature description should follow the format `name1.col1.embedding_size1,name2.col2.embedding_size2,...`. There is no limitied of the features columns.

```sh
FOLDER=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/dev
PROGRAM=/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v4
PREEMB=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/pre-trained/w2v.100 
WORDDIM=100
OUTPUT=best_pos30_chunk30_w2v100_iobes.txt
TAGSCHEME=iobes
PREFIX=pos30chunk30
FEATURE=pos.1.30,chunk.2.30


python $PROGRAM/train.py --tag_scheme iobes --word_dim $WORDDIM --best_outpath $OUTPUT --pre_emb $PREEMB --train $FOLDER/train.conll  --dev $FOLDER/val.conll --test $FOLDER/test.conll --lower 0 --zeros 0 --char_dim 25 --cap_dim 3 --lr_method sgd-lr_.002 --word_bidirect 1 --external_features $FEATURE --prefix=$PREFIX --reload 0 --epoch 120
```

The training script will automatically give a name to the model and store it in ./models/

There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc). To see all parameters, simply run:

```
./train.py --help
```


### Predict an unannotated file 

Program: `predict_file.py`


