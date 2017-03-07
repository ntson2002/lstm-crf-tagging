#!/usr/bin/env python

import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils import create_input2
import loader

from utils import models_path, evaluate, eval_script, eval_temp
from loader import word_mapping, char_mapping, tag_mapping, feature_mapping
from loader import update_tag_scheme, prepare_dataset2
from loader import augment_with_pretrained
from model import Model


np.random.seed(1234)

pre_emb_default = ""
data_folder   = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/new-1layer"
train_default = data_folder + "/fold.0.train.conll"
dev_default   = data_folder + "/fold.0.dev.conll"
test_default  = data_folder + "/fold.0.test.conll"
default_features = "pos.1.3,chunk.2.3,wh.3.3,if.4.3,s.5.3"

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default=train_default,
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default=dev_default,
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default=test_default,
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="1",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="0", #25
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="50",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="50",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default=pre_emb_default,
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)

optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.002",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)

optparser.add_option(
    "-x", "--best_outpath", default="best_result.txt",
    help="best result"
)

optparser.add_option(
    "-e", "--external_features", default=default_features,
    help="external features: name:column:dim,name:column:dim"
)

optparser.add_option(
    "-u", "--prefix", default="",
    help=""
)

optparser.add_option(
    "-E", "--epoch", default="100",
    type='int',
    help="number of epoch"
)

opts = optparser.parse_args()[0]
features = []
if opts.external_features != "None":
    features_list = opts.external_features.split(",")
    features = [ {'name': y[0], 'column': int(y[1]), 'dim': int(y[2])} for y in [x.split('.') for x in features_list]]

print features
# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim

parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method
parameters['prefix'] = opts.prefix

# for f in features:
#     parameters[f['name']] = int(f['dim'])

parameters['external_features'] = opts.external_features

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
model = Model(parameters=parameters, models_path=models_path)
print "Model location: %s" % model.model_path

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)


# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]

    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
print "len(dico_chars): ", len(dico_chars)
print "len(char_to_id): ", len(char_to_id)
print "len(id_to_char): ", len(id_to_char)

dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
print "dico_tags: ", dico_tags
print "tag_to_id: ", tag_to_id
print "id_to_tag: ", id_to_tag

feature_maps = []
for f in features:
    print "--------------------"
    dico_ftag, ftag_to_id, id_to_ftag = feature_mapping(train_sentences, f)
    print 'feature_name:', f['name']
    print 'dico_ftag   :', len(dico_ftag)
    print 'ftag_to_id  :', len(ftag_to_id)
    print 'id_to_ftag  :', len(id_to_ftag)
    feature_maps.append(
        {
            'name'      : f['name'],
            'column'    : f['column'],
            'dim'       : f['dim'],
            'dico_ftag' : dico_ftag,
            'ftag_to_id': ftag_to_id,
            'id_to_ftag': id_to_ftag
        }
    )

train_data = prepare_dataset2(
    train_sentences, word_to_id, char_to_id, tag_to_id, feature_maps, lower
)

print "--------"
from pprint import pprint
pprint (train_data[0])
print "--------"
dev_data = prepare_dataset2(
    dev_sentences, word_to_id, char_to_id, tag_to_id, feature_maps, lower
)
test_data = prepare_dataset2(
    test_sentences, word_to_id, char_to_id, tag_to_id, feature_maps, lower
)

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

# Save the mappings to disk
print 'Saving the mappings to disk...'
model.save_mappings(id_to_word, id_to_char, id_to_tag, feature_maps)

# Build the model
# f_train, f_eval, f_test = model.build(**parameters)
f_train, f_eval = model.build(**parameters)

# Reload previous model values
if opts.reload:
    print 'Reloading previous model...'
    model.reload()

#
# Train network
#
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])
n_epochs = opts.epoch  # number of epochs over the training set
freq_eval = 1000  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_test = -np.inf
best_epoch = -1
count = 0

s_result_dev_BEST = ""
s_result_test_SAVED = ""
iob_dev_score_BEST = -np.inf
iob_test_score_BEST = -np.inf
iob_epoch_BEST = -1
eval_lines_best = []

print "model: ", model.model_path
test_score_at_best_dev = 0

if opts.reload:
    print "Reload model and evaluating ............ "
    dev_score, iob_dev_score, s_result_dev, _, _ = evaluate(parameters, f_eval, dev_sentences,
                                                         dev_data, id_to_tag)
    print s_result_dev
    print "##### EVALUATE on TEST"
    test_score, iob_test_score, s_result_test, eval_lines, _ = evaluate(parameters, f_eval, test_sentences,
                                                            test_data, id_to_tag)
    print s_result_test
    test_score_at_best_dev = test_score
    eval_lines_best = eval_lines



for epoch in xrange(n_epochs):
    epoch_costs = []
    print "------------------------------------------------------------------------------------"
    print "Starting epoch %i..." % epoch
    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        input = create_input2(train_data[index], parameters, True, singletons)

        new_cost = f_train(*input)
        epoch_costs.append(new_cost)
        if i % 50 == 0 and i > 0 == 0:
            print "%i, cost average: %f" % (i, np.mean(epoch_costs[-50:]))
        # print "###components: ", model.components.keys()
        # print "TEST !!!!!", f_test(*input)
        if count % freq_eval == 0 or i == len(train_data) - 1:
            print "##### EVALUATE on DEV"
            dev_score, iob_dev_score, s_result_dev, _, _ = evaluate(parameters, f_eval, dev_sentences,
                                 dev_data, id_to_tag)

            print s_result_dev
            print "##### EVALUATE on TEST"
            test_score, iob_test_score, s_result_test, eval_lines, _ = evaluate(parameters, f_eval, test_sentences,
                                  test_data, id_to_tag)

            print s_result_test
            print "Score on dev: %.5f (IOB: %.5f)" % (dev_score, iob_dev_score)
            print "Score on test: %.5f (IOB: %.5f)" % (test_score, iob_test_score)

            if dev_score > best_dev:
                best_dev = dev_score
                best_epoch = epoch
                test_score_at_best_dev = test_score
                eval_lines_best = eval_lines

                print "New best score on dev."
                print "Saving model to disk..."
                model.save()

            if test_score > best_test:
                best_test = test_score
                print "New best score on test."

            if iob_dev_score > iob_dev_score_BEST:
                iob_epoch_BEST = epoch
                iob_dev_score_BEST = iob_dev_score
                s_result_dev_BEST = s_result_dev
                s_result_test_SAVED = s_result_test

    print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))
    with open(opts.best_outpath, "w") as f_out:
        f_out.write("# IOB BEST DEV SCORE at EPOCH: {}\n".format(best_epoch))
        f_out.write(s_result_dev_BEST)

        f_out.write("\n\n# IOB TEST SCORE at best score of DEV: \n")
        f_out.write(s_result_test_SAVED)

        f_out.write("\n")
        if len(eval_lines_best) > 0:
            f_out.write("# TEST EVAL: " + eval_lines_best[0] + "\n")
        f_out.write("# BEST DEV SCORE:  {} at epoch {}\n".format(best_dev, best_epoch))
        f_out.write("# TEST SCORE:  {} at epoch {}\n".format(test_score_at_best_dev, best_epoch))


with open(opts.best_outpath, "w") as f_out:
    f_out.write("# IOB BEST DEV SCORE at EPOCH: {}\n".format(best_epoch))
    f_out.write(s_result_dev_BEST)

    f_out.write("\n\n# IOB TEST SCORE at best score of DEV: \n")
    f_out.write(s_result_test_SAVED)

    f_out.write("\n")
    if len(eval_lines_best) > 0:
        f_out.write("# TEST EVAL: " + eval_lines_best[0] + "\n")
    f_out.write("# BEST DEV SCORE:  {} at epoch {}\n".format(best_dev, best_epoch))
    f_out.write("# TEST SCORE:  {} at epoch {}\n".format(test_score_at_best_dev, best_epoch))