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

Program: `predict_file.py`

Input file `testb2_sample_notag.conll`: The same format with the training data file but the tag at ground truth column are `O` . 

```
 col_0        col_1  col_2  col_3  col_4 
-----------------------------------------
 Bí_thư       N      B-NP   O      O     
 Đảng_uỷ      N      B-NP   O      O     
 phường       N      I-NP   O      O     
 Nghi_Hải     NNP    I-NP   O      O     
 (            CH     O      O      O     
 Cửa_Lò       NNP    B-NP   O      O     
 ,            CH     O      O      O     
 Nghệ_An      NNP    B-NP   O      O     
 )            CH     O      O      O     
 Nguyễn       NNP    B-NP   O      O     
 Văn          NNP    I-NP   O      O     
 Bích         NNP    I-NP   O      O     
 đã           R      O      O      O     
 quả_quyết    V      B-VP   O      O     
 với          E      B-PP   O      O     
 chúng_tôi    P      B-NP   O      O     
 như_vậy      X      O      O      O     
 .            CH     O      O      O     
                                         
 Ông          Ns     B-NP   O      O     
 Nguyễn       NNP    B-NP   O      O     
 Thanh        NNP    I-NP   O      O     
 Hoà          NNP    I-NP   O      O     
 ,            CH     O      O      O     
 cục_trưởng   N      B-NP   O      O     
 Cục          N      B-NP   O      O     
 Quản_lý      V      I-NP   O      O     
 lao_động     N      I-NP   O      O     
 ngoài        N      I-NP   O      O     
 nước         N      I-NP   O      O     
 (            CH     O      O      O     
 Bộ           N      B-NP   O      O     
 Lao_động     N      I-NP   O      O     
 -            CH     I-NP   O      O     
 thương_binh  N      I-NP   O      O     
 &            CH     I-NP   O      O     
 xã_hội       N      I-NP   O      O     
 )            CH     O      O      O     
 ,            CH     O      O      O     
 cũng         R      O      O      O     
 chung        A      B-AP   O      O     
 quan_điểm    N      B-NP   O      O     
 này          P      B-NP   O      O     
 ...          CH     O      O      O     
```

Predict a file:

``` sh

TESTFILE=testb2_sample_notag.conll  # path of test file 
PROGRAM=/home/s1520203/Bitbucket/lstm-crf-tagging/multi-lstm # program 
MODEL=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/run-multi-lstm/models/vn_p30c30 # saved model 
OUTPUT=testb2_sample_predicted_tag.conll # path of output 

python $PROGRAM/predict_file.py --test_file $TESTFILE --out_file $OUTPUT --model $MODEL

```

Output: `testb2_sample_predicted_tag.conll`

```
 Bí_thư       O      O     
 Đảng_uỷ      B-ORG  O     
 phường       I-ORG  B-LOC 
 Nghi_Hải     I-ORG  I-LOC 
 (            O      O     
 Cửa_Lò       B-LOC  O     
 ,            O      O     
 Nghệ_An      B-LOC  O     
 )            O      O     
 Nguyễn       B-PER  O     
 Văn          I-PER  O     
 Bích         I-PER  O     
 đã           O      O     
 quả_quyết    O      O     
 với          O      O     
 chúng_tôi    O      O     
 như_vậy      O      O     
 .            O      O     
                           
 Ông          O      O     
 Nguyễn       B-PER  O     
 Thanh        I-PER  O     
 Hoà          I-PER  O     
 ,            O      O     
 cục_trưởng   O      O     
 Cục          B-ORG  O     
 Quản_lý      I-ORG  O     
 lao_động     I-ORG  O     
 ngoài        I-ORG  O     
 nước         I-ORG  O     
 (            O      O     
 Bộ           B-ORG  O     
 Lao_động     I-ORG  O     
 -            I-ORG  O     
 thương_binh  I-ORG  O     
 &            I-ORG  O     
 xã_hội       I-ORG  O     
 )            O      O     
 ,            O      O     
 cũng         O      O     
 chung        O      O     
 quan_điểm    O      O     
 này          O      O     
 ...          O      O     
```
