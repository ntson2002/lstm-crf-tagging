#!/usr/bin/env python

import os
import codecs
import optparse
import loader
from utils import models_path, evaluate, eval_temp, eval_path
from loader import prepare_dataset
from loader import update_tag_scheme
from model import Model

default_model = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v2-dev/models/tag_scheme=iob,lower=False,zeros=False,char_dim=25,char_lstm_dim=25,char_bidirect=True,word_dim=100,word_lstm_dim=100,word_bidirect=True,pre_emb=train.txt.w2vec100,all_emb=False,cap_dim=3,pos_dim=30,chunk_dim=30,crf=True,dropout=0.5,lr_method=sgd-lr_.002"
default_prefix = ""
default_log = False
# Read parameters from command line
optparser = optparse.OptionParser()

optparser.add_option(
    "-t", "--test_folder", default="./data/ner/test",
    help="Test set location"
)

optparser.add_option(
    "-o", "--out_folder", default="./data/ner/test-result",
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
word_to_id, char_to_id, tag_to_id, pos_to_id, chunk_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag, model.id_to_pos, model.id_to_chunk]
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

        test_data = prepare_dataset(
            test_sentences, word_to_id, char_to_id, tag_to_id, pos_to_id, 1, chunk_to_id, 2, lower=lower
        )

        print "input: ", test_file, ":" , len(test_sentences), len(test_data)
        score_file = opts.out_folder + "/" + opts.prefix + out_file
        print "output: ", score_file

        test_score, iob_test_score, s_result_test, eval_lines, log = evaluate(parameters, f_eval, test_sentences,
                                  test_data, model.id_to_tag, blog=opts.log, eval_script=eval_script)


        with codecs.open(score_file, "w", "utf-8") as f:

            f.write("--------------------------------------------------------")
            f.write("\nTest file: " + opts.test_folder + "/" + test_file);
            f.write("\n--------------------------------------------------------")
            f.write("\n")
            keys = ["#", "RUN"]
            values = ["#", opts.prefix + out_file]
            for k, v in model.parameters.items():
                keys.append(str(k))
                values.append(str(v))
            if eval_script.endswith("conlleval2"):
                temp = eval_lines[3].split("\t")
                keys.extend(["Precision", "Recall", "FB1"])
                values.extend(temp[1:])

            f.write("\t".join(keys) + "\n")
            f.write("\t".join(values) + "\n")

            # paras = ["\t" + str(k) + ":\t" + str(v) for k, v in model.parameters.items()]
            # f.write("\n".join(paras))

            f.write("\n--------------------------------------------------------\n")
            for line in eval_lines:
                f.write(line + "\n")

            if (opts.show_detail):
                f.write("\n--------------------------------------------------------\n")
                f.write(s_result_test + "\n")

            if (opts.log):
                f.write("\n***\n")
                f.write("\n".join(log))


        print s_result_test
        print "Score on test: %.5f (IOB: %.5f)" % (test_score, iob_test_score)
