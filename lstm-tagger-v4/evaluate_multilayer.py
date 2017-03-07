from theano.typed_list.tests.test_basic import test_append

from tools import common as common
import codecs
import os
from utils import zero_digits, call_conlleval
from model import Model
import loader
from utils import models_path, predict, eval_temp, eval_path
from loader import prepare_dataset2
from loader import update_tag_scheme

# step 1:
def save_all_lines(lines, output):
    # print lines
    with codecs.open(output, "w", "utf-8") as f:
        f.write(unicode("\n".join(lines), "utf-8"))


def predict_a_file(test_file, out_file, model, add_o_tag):
    assert os.path.isfile(test_file)

    model = Model(model_path=model)
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

    test_sentences = loader.load_sentences(test_file, lower, zeros)
    update_tag_scheme(test_sentences, tag_scheme)

    test_data = prepare_dataset2(
        test_sentences, word_to_id, char_to_id, tag_to_id, model.feature_maps, lower
    )

    print "input: ", test_file, ":", len(test_sentences), len(test_data)
    print "output: ", out_file

    predict(parameters, f_eval, test_sentences, test_data, model.id_to_tag, out_file, add_O_tags=add_o_tag)


def evaluate(test_file, models, tmp_folder, result_file):
    """
    :param test_file: conll format contains gold label of all layers,
    :param models: list of model path
    :param tmp_folder: temp folder
    :param result_file: result file
    :return:
    """
    input = test_file
    file_name = os.path.basename(test_file)
    lines = common.get_all_lines(input)
    newlines = []
    nlayer = len(models)
    for line in lines:
        if line.strip() != "":
            newline = "\t".join(line.strip().split("\t")[:-nlayer] + ["O"]) # remove all gold label + append O tag
            newlines.append(newline)
        else:
            newlines.append("")

    input_files = [tmp_folder + "/" + file_name + ".temp.layer" + str(i) for i in range(nlayer + 1)]
    eval_files = [tmp_folder + "/" + file_name + ".eval" + str(i) for i in range(nlayer)]
    result_files = [tmp_folder + "/" + file_name + ".result.layer" + str(i) + ".txt" for i in range(nlayer)]

    save_all_lines(newlines, input_files[0]) # create input file of layer 0

    eval_lines_all = []
    nlayer = len(models)
    print "#LAYER:", nlayer
    for i in range(nlayer):

        predict_a_file(input_files[i],  input_files[i + 1], models[i], True)

        lines_i = common.get_all_lines(input_files[i + 1])
        print "#LEN: ", len(lines_i)

        # output conll to conlleval
        lines_eval = []
        for k in range(len(lines)):
            if lines[k].strip() != "":
                tokens1 = lines[k].strip().split("\t")
                tokens2 = lines_i[k].strip().split("\t")
                assert zero_digits(tokens1[0]) == zero_digits(tokens2[0])
                lines_eval.append(" ".join([tokens1[0], tokens1[i - nlayer], tokens2[-2]]))
            else:
                lines_eval.append("")

        save_all_lines(lines_eval, eval_files[i])
        eval_lines = call_conlleval(eval_files[i], result_files[i])
        eval_lines_all.append("=======================")
        eval_lines_all.append("LAYER: " + str(i))
        eval_lines_all.extend(eval_lines)

    result_file.write("\n".join(eval_lines_all))



def main_old():
    def evaluate_a_run_folder(model_folder, result_file_path, test_folder, temp, nfold):
        # model_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-v1/01jpc-emb-crf-d100/00emb-crf-nofeature-lr002"
        # result_file_path = model_folder + "/00emb-crf-nofeature-lr002.txt"
        # test_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/test-3layer"
        # temp = model_folder + "/out-temp"
        # nfold = 10
        # nlayer = 3

        result_file = codecs.open(result_file_path, "w", "utf-8")

        for i in range(nfold):
            test_file = "fold.%d.test.conll" % (i)
            # print test_file

            models = [
                model_folder + "/layer0/models/prefix=%d" % (i),  # layer 0
                model_folder + "/layer1/models/prefix=%d" % (i),  # layer 1
                # model_folder + "/layer2/models/prefix=%d" % (i)
            ]
            assert os.path.isfile(test_folder + "/" + test_file)
            assert os.path.isdir(models[0])
            assert os.path.isdir(models[1])
            # assert os.path.isdir(models[2])
            result_file.write(
                "\n-------------------------------------------------------------------------------------\n")
            result_file.write("FOLD %i :\n" % (i))
            result_file.write("test file: " + test_folder + "/" + test_file + "\n")
            result_file.write("model layer 0: " + models[0] + "\n")
            result_file.write("model layer 1: " + models[1] + "\n")
            # result_file.write("model layer 2: " + models[2] + "\n")
            evaluate(test_folder + "/" + test_file, models, temp, result_file)

    model_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/run-separate-layer/01emb-crf-d100"
    result_file_path = model_folder + "/01emb-crf-d100.txt"
    test_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/data/conll-2layer"
    temp = model_folder + "/out-temp"
    nfold = 1

    # model_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-v1/01jpc-emb-crf-d100/00emb-crf-nofeature-lr002"
    # result_file_path = model_folder + "/00emb-crf-nofeature-lr002.txt"
    # test_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/test-3layer"
    # temp = model_folder + "/out-temp"
    # nfold = 1

    evaluate_a_run_folder(model_folder, result_file_path, test_folder, temp, nfold)


if __name__ == '__main__':
    import optparse
    optparser = optparse.OptionParser()

    optparser.add_option(
        "-t", "--test_file",
        default="",
        help="Test file"
    )

    optparser.add_option(
        "-m", "--models",
        default="",
        help="List all model paths, seperated by #"
    )

    optparser.add_option(
        "-x", "--temp_folder",
        default="",
        help="Temp folder that will contain temporpary files"
    )

    optparser.add_option(
        "-o", "--result_path",
        default="",
        help="Path of result file"
    )

    opts = optparser.parse_args()[0]
    print opts
    models = opts.models.split("#")

    with codecs.open(opts.result_path, "a", "utf-8") as result_file:
        result_file.write("\n-------------------------------------------------------------------------------------\n")
        result_file.write("test file: " + opts.test_file + "\n")
        for i in range(len(models)):
            result_file.write("model layer %d: %s\n"%(i, models[i]))

        evaluate(opts.test_file, models, opts.temp_folder, result_file)
