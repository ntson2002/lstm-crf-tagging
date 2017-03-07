#!/usr/bin/env python

"""

"""
import os
import optparse
import loader
from utils import models_path, predict
from loader import prepare_dataset2
from loader import update_tag_scheme
from model import Model

default_model = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-v1/01jpc-emb-crf-d100/06emb-crf-f12345-lr002/layer0/models/prefix=0"
default_prefix = "predict-"
default_log = False

optparser = optparse.OptionParser()

optparser.add_option(
    "-t", "--test_folder", default="/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/test-1layer",
    help="Test set location"
)

optparser.add_option(
    "-o", "--out_folder", default="/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-v1/01jpc-emb-crf-d100/06emb-crf-f12345-lr002/layer0/prefix=0",
    help="Output location"
)

optparser.add_option(
    "-m", "--model", default=default_model,

    help="Model location"
)

optparser.add_option(
    "-p", "--prefix", default=default_prefix,

    help="Prefix of result file"
)

opts = optparser.parse_args()[0]

# Check parameters validity
assert not os.path.isfile(opts.test_folder)

if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
model = Model(model_path=opts.model)
parameters = model.parameters

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = model.parameters['tag_scheme']

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]


print 'Reloading previous model...'
_, f_eval = model.build(training=False, **parameters)
model.reload()


print "--------"
print opts.test_folder
for test_file in os.listdir(opts.test_folder):

    if test_file.endswith(".conll"):
        out_file = default_prefix + test_file.replace(".conll", "") + ".txt"

        test_sentences = loader.load_sentences(opts.test_folder + "/" + test_file, lower, zeros)
        
        update_tag_scheme(test_sentences, tag_scheme)

        test_data = prepare_dataset2(
            test_sentences, word_to_id, char_to_id, tag_to_id, model.feature_maps, lower
        )

        print "input: ", test_file, ":", len(test_sentences), len(test_data)
        print "output: ", opts.out_folder + "/" + out_file

        predict(parameters, f_eval, test_sentences,
                test_data, model.id_to_tag, opts.out_folder + "/" + out_file, add_O_tags=False)