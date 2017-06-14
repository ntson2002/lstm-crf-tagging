# Nested named entity recognition using multilayer recurrent neural networks

This model is reported in the paper: Nguyen Truong Son, Nguyen Le Minh, "*Nested named entity recognition using multilayer recurrent neural networks*", PACLING 2017, August 16 - 18, 2017, Sedona Hotel, Yangon, Myanmar (to be appear)


Requirements:

*  Python 2.7, with Numpy and Theano installed.


Two implemented models:

* [lstm-tagger-v4](https://github.com/ntson2002/lstm-crf-tagging/tree/master/lstm-tagger-v4): Implementation of single BI-LSTM-CRF with additional features to recognize named entites at the top level.  

* [multi-lstm](https://github.com/ntson2002/lstm-crf-tagging/tree/master/multi-lstm): Implementation of Multilayer BI-LSTM-CRF model to recognize nested named entities.


Our proposed models are based on [Lample et al 2016](https://arxiv.org/abs/1603.01360).



# Evaluation results on the official test set VLSP 2016
* VLSP 2016: <http://vlsp.org.vn/evaluation_campaign_NER>
* Results on the official test set 

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
