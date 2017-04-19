# Nested named entity recognition using multilayer recurrent networks

Requirements:

*  Python 2.7, with Numpy and Theano installed.


Two implemented models:

* [lstm-tagger-v4](https://github.com/ntson2002/lstm-crf-tagging/tree/master/lstm-tagger-v4): Implementation of single BI-LSTM-CRF with additional features.  

* [multi-lstm](https://github.com/ntson2002/lstm-crf-tagging/tree/master/multi-lstm): Implementation of Multilayer BI-LSTM-CRF model 


Our proposed models are based on [Lample et al 2016](https://arxiv.org/abs/1603.01360). But the orginal model do not focus to additional features as well as it can only recognize named entities at 1 layer.
