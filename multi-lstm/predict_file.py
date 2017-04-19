#!/usr/bin/env python

import os
import optparse
import loader
from utils import predict_multilayer
from loader import prepare_dataset3
from loader import update_tag_scheme
from model import Model

# default_model = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/run-multi-lstm/models/tag_scheme=iobes,char_dim=25,word_dim=100,word_bidirect=True,pre_emb=train.txt.w2vec100,all_emb=False,crf=True,dropout=0.5,external_features=pos.1.30chunk.2.30,prefix=p30c30"
default_model = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/run-multi-lstm/models/vn_p30c30"
default_test_file = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/data/conll-2layer/testb2_notag.conll"
default_out_file = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/run-multi-lstm/testb2222.conll.txt"
default_prefix = ""

default_log = False

# Read parameters from command line
optparser = optparse.OptionParser()

optparser.add_option(
    "-t", "--test_file", default=default_test_file,
    help="Test set location (conll format)"
)

optparser.add_option(
    "-o", "--out_file", default=default_out_file,
    help="Output location"
)

optparser.add_option(
    "-m", "--model", default=default_model,
    help="Model location"
)

optparser.add_option(
    "-a", "--add_o_tag", default=True,
    help="Add a column O at the end"
)

opts = optparser.parse_args()[0]

# Initialize model
model = Model(model_path=opts.model)
parameters = model.parameters

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = model.parameters['tag_scheme']

# Load reverse mappings
word_to_id, char_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char]
]

print 'Reloading previous model...'
_, f_eval = model.build(training=False, **parameters)
model.reload()


assert os.path.isfile(opts.test_file)
test_file = opts.test_file
out_file = opts.out_file
test_sentences = loader.load_sentences(test_file, lower, zeros)
update_tag_scheme(test_sentences, tag_scheme)

test_data = prepare_dataset3(
    test_sentences, word_to_id, char_to_id, model.tag_maps, model.feature_maps, lower
)

# print test_data[0]
print "input: ", test_file, ":", len(test_sentences), len(test_data)
print "output: ", out_file

predict_multilayer(parameters, f_eval, test_sentences, test_data, model.tag_maps, out_file)



