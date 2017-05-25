# Nested named entity recognition using multilayer recurrent neural networks

This model is reported in the paper: Nguyen Truong Son, Nguyen Le Minh, "*Nested named entity recognition using multilayer recurrent neural networks*", PACLING 2017, August 16 - 18, 2017, Sedona Hotel, Yangon, Myanmar (to be appear)

Requirements:

*  Python 2.7, with Numpy and Theano installed.


Two implemented models:

* [lstm-tagger-v4](https://github.com/ntson2002/lstm-crf-tagging/tree/master/lstm-tagger-v4): Implementation of single BI-LSTM-CRF with additional features to recognize named entites at the top level.  

* [multi-lstm](https://github.com/ntson2002/lstm-crf-tagging/tree/master/multi-lstm): Implementation of Multilayer BI-LSTM-CRF model to recognize nested named entities.


Our proposed models are based on [Lample et al 2016](https://arxiv.org/abs/1603.01360). But the orginal model does not focus to additional features as well as recognizing nested named entities.
