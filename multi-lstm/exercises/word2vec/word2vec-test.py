#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim

# model = gensim.models.Word2Vec.load_word2vec_format('/home/s1520203/programs/word2vec/output/du-lieu-luat-all-tokenized.dat-100.bin', binary=True, unicode_errors='ignore')
# model = gensim.models.Word2Vec.load_word2vec_format('/home/s1520203/programs/word2vec/output/tokenize.vi.256.bin', binary=True, unicode_errors='ignore')
model = gensim.models.Word2Vec.load_word2vec_format('/home/s1520203/programs/word2vec/output/all-data-tokenized.vi-100.bin', binary=True, unicode_errors='ignore')
# model = gensim.models.Word2Vec.load_word2vec_format('/work/sonnguyen/glove/glove_word2vec/glove.6B.100d.w2vec', binary=False, unicode_errors='ignore')
# model = gensim.models.Word2Vec.load_word2vec_format('/work/sonnguyen/glove/glove_word2vec/glove.twitter.27B.100d.w2vec', binary=False, unicode_errors='ignore')
# print len(model)
w = model[u"man"]
print type(w)
print len(w)

# print model.similarity(u'đường_bộ', u'đường_sắt')

list = model.similar_by_word(u'hạt_nhân', topn=30)
for word in list:
        print word[0], word[1]
