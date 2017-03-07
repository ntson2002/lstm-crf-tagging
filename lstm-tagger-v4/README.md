#lstm-tagger-v4

## BI-LSTM-CRF with features

BI-LSTM-CRF Tagger is an implementation of a Named Entity Recognizer that obtains state-of-the-art performance in NER on the 4 CoNLL datasets (English, Spanish, German and Dutch) without resorting to any language-specific knowledge or resources such as gazetteers. Details about the model can be found at: http://arxiv.org/abs/1603.01360



## Initial setup

To use the tagger, you need Python 2.7, with Numpy and Theano installed.


## Train a model

To train your own model, you need to use the train.py script and provide the location of the training, development and testing set:

```
FOLDER=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/new-3layer
PROGRAM=/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v3

EPOCH=200
TYPE=sgd-lr_.002
TAGSCHEME=iobes
CAPDIM=0
ZERO=1
LOWER=0
WORDDIM=100
WORDLSTMDIM=100
CRF=1
RELOAD=0
CHARDIM=0


FEATURE=pos.1.10,layer1.7.10,layer2.8.10
ALLEMB=1
PREEMB=/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/pretrained/rre.w2v100.txt


###########
FOLD=0
BESTFILE=best.$FOLD.txt
python $PROGRAM/train.py --char_dim $CHARDIM --word_lstm_dim $WORDLSTMDIM --train $FOLDER/fold.$FOLD.train.conll --dev $FOLDER/fold.$FOLD.dev.conll --test $FOLDER/fold.$FOLD.test.conll --best_outpath $BESTFILE --lr_method $TYPE --word_dim $WORDDIM --tag_scheme $TAGSCHEME --cap_dim $CAPDIM --zeros $ZERO --lower $LOWER --reload $RELOAD --external_features $FEATURE --epoch $EPOCH --crf $CRF --pre_emb $PREEMB --prefix=$FOLD

```

The training script will automatically give a name to the model and store it in ./models/
There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc). To see all parameters, simply run:

```
./train.py --help
```

Input files for the training script have to follow the same format than the CoNLL2003 sharing task: each word has to be on a separate line, and there must be an empty line after each sentence. A line must contain at least 2 columns, the first one being the word itself, the last one being the named entity. It does not matter if there are extra columns that contain tags or chunks in between. Tags have to be given in the IOB format (it can be IOB1 or IOB2).

### Reproduce the result
```
np.random.seed(1234)
```
### Japanese Civil Code RRE corpus

```
If	IN	O	O	B-IF	O	SBAR/S/	B-R	O	O
the	DT	B-NP	O	I-IF	B-S	NP/S/SBAR/S/	I-R	O	O
prescription	NN	E-NP	O	I-IF	I-S	NP/S/SBAR/S/	I-R	O	O
may	MD	O	O	I-IF	I-S	VP/S/SBAR/S/	I-R	O	O
not	RB	O	O	I-IF	I-S	VP/S/SBAR/S/	I-R	O	O
be	VB	O	O	I-IF	I-S	VP/S/SBAR/S/	I-R	O	O
interrupted	VBN	O	O	I-IF	I-S	VP/S/SBAR/S/	I-R	O	O
upon	IN	O	O	I-IF	I-S	PP/VP/S/SBAR/S/	I-R	O	O
expiration	NN	B-NP	O	I-IF	I-S	NP/PP/VP/S/SBAR/S/	I-R	O	O
of	IN	I-NP	O	I-IF	I-S	PP/NP/PP/VP/S/SBAR/S/	I-R	O	O
period	NN	I-NP	O	I-IF	I-S	NP/PP/NP/PP/VP/S/SBAR/S/	I-R	O	O
of	IN	I-NP	O	I-IF	I-S	PP/NP/PP/NP/PP/VP/S/SBAR/S/	I-R	O	O
the	DT	I-NP	O	I-IF	I-S	NP/PP/NP/PP/NP/PP/VP/S/SBAR/S/	I-R	O	O
prescription	NN	E-NP	O	I-IF	I-S	NP/PP/NP/PP/NP/PP/VP/S/SBAR/S/	I-R	O	O
due	JJ	O	O	I-IF	I-S	PP/VP/S/SBAR/S/	I-R	O	O
to	TO	O	O	I-IF	I-S	PP/VP/S/SBAR/S/	I-R	O	O
any	DT	B-NP	O	I-IF	I-S	NP/PP/VP/S/SBAR/S/	I-R	O	O
natural	JJ	I-NP	O	I-IF	I-S	NP/PP/VP/S/SBAR/S/	I-R	O	O
disaster	NN	I-NP	O	I-IF	I-S	NP/PP/VP/S/SBAR/S/	I-R	O	O
or	CC	I-NP	O	I-IF	I-S	NP/PP/VP/S/SBAR/S/	I-R	O	O
other	JJ	I-NP	O	I-IF	I-S	NP/PP/VP/S/SBAR/S/	I-R	O	O
unavoidable	NN	I-NP	O	I-IF	I-S	NP/PP/VP/S/SBAR/S/	I-R	O	O
contingency	NN	E-NP	O	E-IF	E-S	NP/PP/VP/S/SBAR/S/	I-R	O	O
,	,	O	O	O	O	S/	O	O	O
the	DT	B-NP	O	O	O	NP/S/	B-E	O	O
prescription	NN	E-NP	O	O	O	NP/S/	I-E	O	O
shall	MD	O	O	O	O	VP/S/	I-E	O	O
not	RB	O	O	O	O	VP/S/	I-E	O	O
be	VB	O	O	O	O	VP/S/	I-E	O	O
completed	VBN	O	O	O	O	VP/S/	I-E	O	O
until	IN	O	O	O	O	PP/VP/S/	B-R	O	O
two	CD	B-NP	O	O	O	NP/PP/VP/S/	I-R	O	O
weeks	NNS	E-NP	O	O	O	NP/PP/VP/S/	I-R	O	O
elapse	JJ	O	O	O	O	ADJP/NP/PP/VP/S/	I-R	O	O
from	IN	O	O	O	O	PP/ADJP/NP/PP/VP/S/	I-R	O	O
the	DT	B-NP	O	O	O	NP/PP/ADJP/NP/PP/VP/S/	I-R	O	O
time	NN	E-NP	O	O	O	NP/PP/ADJP/NP/PP/VP/S/	I-R	O	O
when	WRB	O	O	B-IF	O	WHADVP/SBAR/NP/PP/ADJP/NP/PP/VP/S/	I-R	O	O
such	JJ	B-NP	O	I-IF	O	NP/S/SBAR/NP/PP/ADJP/NP/PP/VP/S/	I-R	O	O
impediment	NN	E-NP	O	I-IF	O	NP/S/SBAR/NP/PP/ADJP/NP/PP/VP/S/	I-R	O	O
has	VBZ	B-VP	O	I-IF	O	VP/S/SBAR/NP/PP/ADJP/NP/PP/VP/S/	I-R	O	O
ceased	VBN	I-VP	O	I-IF	O	VP/S/SBAR/NP/PP/ADJP/NP/PP/VP/S/	I-R	O	O
to	TO	I-VP	O	I-IF	O	VP/S/VP/S/SBAR/NP/PP/ADJP/NP/PP/VP/S/	I-R	O	O
exist	VB	E-VP	O	E-IF	O	VP/S/VP/S/SBAR/NP/PP/ADJP/NP/PP/VP/S/	I-R	O	O
.	.	O	O	O	O	S/	O	O	O
```
### Japanase Pension Law RRE corpus 


### English NER corpus 


### Vietnamse NER corpus

