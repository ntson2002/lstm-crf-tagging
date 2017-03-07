import os
import re
import codecs
import numpy as np
import theano


models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval_multi")


def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")

def get_path(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    selected_keys = {'tag_scheme', 'word_dim', 'word_bidirect', 'pre_emb', 'crf', 'all_emb', 'external_features', 'dropout', 'char_dim', 'prefix'}
    for k, v in parameters.items():
        if k in selected_keys:
            if type(v) is str and "/" in v:
                l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
            else:
                l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")

def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))


def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """

    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """

    # tagset = {tag for i, tag in enumerate(tags)}
    # print tagset
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    # print singletons
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input_multilayer(data, parameters, add_label, singletons=None):
    """
        Take sentence data and return an input for
        the training or the evaluation function.
        """
    # features = [{'name': y[0], 'column': int(y[1]), 'dim': int(y[2])} for y in
    #             [x.split('.') for x in parameters['external_features'].split(",")]]

    features = parse_feature_string(parameters['external_features'])
    tags = parse_tag_column_string(parameters['tag_columns_string'])

    words = data['words']
    chars = data['chars']

    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []

    if parameters['word_dim']:
        input.append(words)

    for f in features:
        input.append(data[f['name']])

    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_bidirect']:
            input.append(char_rev)
        input.append(char_pos)

    if parameters['cap_dim']:
        input.append(caps)

    if add_label:
        # input.append(data['tags0'])
        # data_object['tags' + str(tm['layer'])] = tags
        # print tags
        for t in tags:
            input.append(data['tags'+str(t['layer'])])

    return input



def evaluate_multilayer(parameters, f_eval, raw_sentences, parsed_sentences, tag_maps,
             blog=False, eval_script=eval_script):
    """
    Evaluate current model using CoNLL script.
    """
    log = []
    predictions_list =  [[] for x in xrange(len(tag_maps))]
    counts = []
    for tm in tag_maps:
        counts.append(np.zeros((len(tm['id_to_tag']), len(tm['id_to_tag'])), dtype=np.int32))


    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        if (blog):
            log.append("")
            log.append("")
            log.append("SENTENCE+\t" + ' '.join(tokens[0] for tokens in raw_sentence))
        input = create_input_multilayer(data, parameters, False)
        if parameters['crf']:
            out =  f_eval(*input)
            y_preds_list = [np.array(x)[1:-1] for x in out]

        else:
            y_preds_list = [x.argmax(axis=1) for x in f_eval(*input)]

        for tm in tag_maps:
            id_to_tag = tm['id_to_tag']
            y_preds = y_preds_list[tm['layer']]
            # n_tags = len(id_to_tag)

            y_reals = np.array(data['tags' + str(tm['layer'])]).astype(np.int32)

            # print "#### len", len(y_preds), len(y_reals)
            assert len(y_preds) == len(y_reals)
            p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
            r_tags = [id_to_tag[y_real] for y_real in y_reals]
            if parameters['tag_scheme'] == 'iobes':
                p_tags = iobes_iob(p_tags)
                r_tags = iobes_iob(r_tags)


            for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
                # print "========="
                # print "raw_sentence[i]"
                # print raw_sentence[i]
                # print tm['layer'], r_tags[i], p_tags[i]
                # print raw_sentence[i][:-1]

                new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
                # print new_line
                # print "========="
                predictions_list[tm['layer']].append(new_line)
                counts[tm['layer']][y_real, y_pred] += 1

            predictions_list[tm['layer']].append("")


    results = []
    for tm in tag_maps:
        predictions = predictions_list[tm['layer']]
        n_tags = len(tm['id_to_tag'])
        eval_id = np.random.randint(1000000, 2000000)
        output_path = os.path.join(eval_temp, "eval.%i.output" % eval_id)

        scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)
        with codecs.open(output_path, 'w', 'utf8') as f:
            f.write("\n".join(predictions))
        os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

        print 'Layer', tm['layer'], ":", scores_path
        eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]

        os.remove(output_path)
        os.remove(scores_path)

        confusion_table = ("{}\t{}\t{}\t%s{}\t{}\t{}\t{}\t{}\t\n" % ("{}\t" * n_tags)).format(
            "ID", "NE", "Total",
            *([tm['id_to_tag'][i] for i in xrange(n_tags)] + ["Predict"] + ["Correct"] + ["Recall"] + ["Precision"]+ ["F1"])
        )
        for i in xrange(n_tags):
            correct = counts[tm['layer']][i][i]
            predict = sum([counts[tm['layer']][j][i] for j in xrange(n_tags)])
            recall = counts[tm['layer']][i][i] * 100. / max(1, counts[tm['layer']][i].sum())
            precision = correct * 100. / max(predict, 1)
            f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            confusion_table += ("{}\t{}\t{}\t%s{}\t{}\t{}\t{}\t{}\t\n" % ("{}\t" * n_tags)).format(
                str(i), tm['id_to_tag'][i], str(counts[tm['layer']][i].sum()),
                *([counts[tm['layer']][i][j] for j in xrange(n_tags)] +
                [predict] +
                [correct] +
                ["%.3f" % (recall)] +
                ["%.3f" % (precision)] +
                ["%.3f" % (f1)]
                  )
            )
        # Global accuracy

        iob_accuracy = 100. * counts[tm['layer']].trace() / max(1, counts[tm['layer']].sum())
        confusion_table +=  "%i/%i (%.5f%%)" % (counts[tm['layer']].trace(), counts[tm['layer']].sum(), iob_accuracy)

        fb1 = float(eval_lines[1].strip().split()[-1])
        result = eval(eval_lines[0])
        result['total_token'] = max(1, counts[tm['layer']].sum())
        result['corrected_token'] = counts[tm['layer']].trace()
        # F1 on all entities
        results.append( {
                        '_layer': tm['layer'],
                        'fb1': fb1,
                        'result': result,
                        'confusion_table': confusion_table,
                        'conlleval' : '\n'.join(eval_lines[1:])
                        }
        )


    temp = np.array([[r['result']['total_phrased'], r['result']['total_found'], r['result']['correct'], r['result']['total_token'], r['result']['corrected_token'] ] for r in results])
    [total_phrased, total_found, correct, total_token, corectted_token ] = temp.sum(axis=0)
    p = correct / float(total_found)
    r = correct / float(total_phrased)
    fb1 = 2 * p * r / (p + r)
    iob_accuracy_all = corectted_token / float(total_token)

    return fb1, iob_accuracy_all, results, log


def parse_feature_string(feature_string):
    return [{'name': y[0], 'column': int(y[1]), 'dim': int(y[2])} for y in
     [x.split('.') for x in feature_string.split(",")]]

def parse_tag_column_string(tags_column_string):
    columns = tags_column_string.split(",")
    tags_list = []
    for layer_index in range(len(columns)):
        tags_list.append(
            {
                'layer': int(layer_index),
                'column': int(columns[layer_index])
            })

    return tags_list

def print_evaluation_result(results):
    print "==================="
    for r in results:
        layer = r['_layer']
        print
        print "*** LAYER: ", layer
        print "- Precision: %.5f Recall: %.5f FB1: %.5f" %(r['result']['precision'], r['result']['recall'], r['result']['FB1'])
        print "- Gold: ", r['result']['total_phrased'], "Found: ", r['result']['total_found'], "Correct: ", r['result']['correct']

        # print "* Confusion matrix:"
        # print r['confusion_table']
        print "* Conlleval result: "
        print r['conlleval']
        # print "==================="





def predict_multilayer(parameters, f_eval, raw_sentences, parsed_sentences, tag_maps, output_path=None):

    """
    predict a file
    n (layer) last columns is dummy columns which contains 0 tag. After predicting, they will be replaced by system prediction
    """
    nlayer = len(tag_maps)
    predictions =  []
    out_sentences = []

    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        input = create_input_multilayer(data, parameters, False)
        if parameters['crf']:
            out =  f_eval(*input)
            y_preds_list = [np.array(x)[1:-1] for x in out]
        else:
            y_preds_list = [x.argmax(axis=1) for x in f_eval(*input)]


        p_tags_list = []
        for tm in tag_maps:
            id_to_tag = tm['id_to_tag']
            y_preds = y_preds_list[tm['layer']]
            p_tags = [id_to_tag[y_pred] for y_pred in y_preds]

            if parameters['tag_scheme'] == 'iobes':
                p_tags = iobes_iob(p_tags)

            p_tags_list.append(p_tags)

        # print p_tags_list
        p_tags_list = np.array(p_tags_list)

        out_sentence = []

        for i in range(len(raw_sentence)):
            # new_line = "\t".join(raw_sentence[i][:-nlayer]) + "\t" + "\t".join(p_tags_list[:, i])
            new_line = raw_sentence[i][0] + "\t" + "\t".join(p_tags_list[:, i])
            predictions.append(new_line)

            # print [raw_sentence[i][0]]
            # print p_tags_list[:, i]

            a = [raw_sentence[i][0]] + list(p_tags_list[:, i])
            out_sentence.append(a)
        predictions.append("")
        out_sentences.append(out_sentence)

    if output_path:
        with codecs.open(output_path, 'w', 'utf8') as f:
            f.write("\n".join(predictions))

    return out_sentences