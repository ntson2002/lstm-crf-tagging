#!/usr/bin/env python

import os
import codecs
import optparse
import loader
from utils import models_path, predict, eval_temp, eval_path
from loader import prepare_dataset2
from loader import update_tag_scheme
from model import Model

default_model = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-scripts/script-1a-emb-crf-f12345-lr002/models/tag_scheme=iobes,word_dim=50,word_bidirect=True,pre_emb=rre.w2v50.txt,all_emb=False,crf=True,dropout=0.5,lr_method=sgd-lr_.002,prefix=0,external_features=pos.1.3chunk.2.3wh.3.3if.4.3s.5.3"
default_prefix = ""
default_log = False

# Read parameters from command line
optparser = optparse.OptionParser()

optparser.add_option(
    "-t", "--test_folder", default="/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/test-1layer",
    help="Test set location"
)

optparser.add_option(
    "-o", "--out_folder", default="/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-scripts/test-results",
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

optparser.add_option(
    "-l", "--log", default=default_log,
    help="Show the result of tagging"
)

optparser.add_option(
    "-e", "--eval_script", default="conlleval",
    help="Show the result of tagging"
)

optparser.add_option(
    "-d", "--show_detail", default=False,
    help="Show IOB table result"
)


opts = optparser.parse_args()[0]

# Check parameters validity
assert not os.path.isfile(opts.test_folder)
eval_script = os.path.join(eval_path, opts.eval_script)

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
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
        out_file = test_file.replace(".conll", "") + ".txt"

        test_sentences = loader.load_sentences(opts.test_folder + "/" + test_file, lower, zeros)
        
        update_tag_scheme(test_sentences, tag_scheme)

        test_data = prepare_dataset2(
            test_sentences, word_to_id, char_to_id, tag_to_id, model.feature_maps, lower
        )

        print "input: ", test_file, ":", len(test_sentences), len(test_data)
        print "output: ", opts.out_folder + "/" + out_file

        predict(parameters, f_eval, test_sentences,
                test_data, model.id_to_tag, opts.out_folder + "/" + out_file)



