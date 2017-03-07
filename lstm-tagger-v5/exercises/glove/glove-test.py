#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import numpy as np

model_folder = "/work/sonnguyen/glove"
pre_emb = model_folder + "/" + "glove.twitter.27B.25d.txt"

pretrained = {}
word_dim = 25
emb_invalid = 0

print 'Loading pretrained embeddings from %s...' % pre_emb

for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
    line = line.rstrip().split()
    if len(line) == word_dim + 1:
        pretrained[line[0]] = np.array(
            [float(x) for x in line[1:]]
        ).astype(np.float32)
    else:
        print line
        emb_invalid += 1

if emb_invalid > 0:
    print 'WARNING: %i invalid lines' % emb_invalid


print len(pretrained)
