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

#### Input file 

The input file has the same format with the training data file. The last columns cotains `O` tags.

Sample file: `input_notag.conll`

```
 Bí_thư       N      B-NP   O     
 Đảng_uỷ      N      B-NP   O     
 phường       N      I-NP   O     
 Nghi_Hải     NNP    I-NP   O     
 (            CH     O      O     
 Cửa_Lò       NNP    B-NP   O     
 ,            CH     O      O     
 Nghệ_An      NNP    B-NP   O     
 )            CH     O      O     
 Nguyễn       NNP    B-NP   O     
 Văn          NNP    I-NP   O     
 Bích         NNP    I-NP   O     
 đã           R      O      O     
 quả_quyết    V      B-VP   O     
 với          E      B-PP   O     
 chúng_tôi    P      B-NP   O     
 như_vậy      X      O      O     
 .            CH     O      O     
                                  
 Ông          Ns     B-NP   O     
 Nguyễn       NNP    B-NP   O     
 Thanh        NNP    I-NP   O     
 Hoà          NNP    I-NP   O     
 ,            CH     O      O     
 cục_trưởng   N      B-NP   O     
 Cục          N      B-NP   O     
 Quản_lý      V      I-NP   O     
 lao_động     N      I-NP   O     
 ngoài        N      I-NP   O     
 nước         N      I-NP   O     
 (            CH     O      O     
 Bộ           N      B-NP   O     
 Lao_động     N      I-NP   O     
 -            CH     I-NP   O     
 thương_binh  N      I-NP   O     
 &            CH     I-NP   O     
 xã_hội       N      I-NP   O     
 )            CH     O      O     
 ,            CH     O      O     
 cũng         R      O      O     
 chung        A      B-AP   O     
 quan_điểm    N      B-NP   O     
 này          P      B-NP   O     
 ...          CH     O      O     
```

Predict tags:

Saved models: `vn_p30c30`

```sh 

python predict_file.py --test_file ./models/sample_data/sample_notag.conll --out_file sample.conll --model ./models/vn_p30c30
```

Output:

```
 Bí_thư       N    B-NP  O     
 Đảng_uỷ      N    B-NP  B-ORG 
 phường       N    I-NP  I-ORG 
 Nghi_Hải     NNP  I-NP  I-ORG 
 (            CH   O     O     
 Cửa_Lò       NNP  B-NP  B-LOC 
 ,            CH   O     O     
 Nghệ_An      NNP  B-NP  B-LOC 
 )            CH   O     O     
 Nguyễn       NNP  B-NP  B-PER 
 Văn          NNP  I-NP  I-PER 
 Bích         NNP  I-NP  I-PER 
 đã           R    O     O     
 quả_quyết    V    B-VP  O     
 với          E    B-PP  O     
 chúng_tôi    P    B-NP  O     
 như_vậy      X    O     O     
 .            CH   O     O     
                               
 Ông          Ns   B-NP  O     
 Nguyễn       NNP  B-NP  B-PER 
 Thanh        NNP  I-NP  I-PER 
 Hoà          NNP  I-NP  I-PER 
 ,            CH   O     O     
 cục_trưởng   N    B-NP  O     
 Cục          N    B-NP  B-ORG 
 Quản_lý      V    I-NP  I-ORG 
 lao_động     N    I-NP  I-ORG 
 ngoài        N    I-NP  I-ORG 
 nước         N    I-NP  I-ORG 
 (            CH   O     O     
 Bộ           N    B-NP  B-ORG 
 Lao_động     N    I-NP  I-ORG 
 -            CH   I-NP  I-ORG 
 thương_binh  N    I-NP  I-ORG 
 &            CH   I-NP  I-ORG 
 xã_hội       N    I-NP  I-ORG 
 )            CH   O     O     
 ,            CH   O     O     
 cũng         R    O     O     
 chung        A    B-AP  O     
 quan_điểm    N    B-NP  O     
 này          P    B-NP  O     
 ...          CH   O     O     
```

# Evaluation results on the official test set VLSP 2016


| Model | POS | CHUNK | Pre-trained | F1 %  |                           |
|-------|-----|-------|-------------|-------|---------------------------|
| 1     |     |       |             | 82.9  | baseline1 (Lample et. al) |
| 2     | X   |       |             | 86.44 | +3.54%                    |
| 3     |     | X     |             | 89.77 | +6.87%                    |
| 4     | X   | X     |             | 90.27 | +7.37%                    |
| 5     |     |       | X           | 86.84 | baseline2 (Lample et. al) |
| 6     | X   |       | X           | 88.66 | +1.82%                    |
| 7     |     | X     | X           | 91.79 | +4.95%                    |
| 8     | X   | X     | X           | 92.97 | +6.13%                    |
