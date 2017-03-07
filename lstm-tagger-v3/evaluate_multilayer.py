from theano.typed_list.tests.test_basic import test_append

from tools import common as common
import codecs
import os
from utils import zero_digits, call_conlleval
# step 1:

def save_all_lines(lines, output):
    # print lines
    with codecs.open(output, "w", "utf-8") as f:
        f.write(unicode("\n".join(lines), "utf-8"))

def evaluate(input_folder, file, models, folder, result_file, nlayer=3):
    # models = {
    #     0: "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-scripts/script-1a-emb-crf-f12345-lr002/models/tag_scheme=iobes,word_dim=50,word_bidirect=True,pre_emb=rre.w2v50.txt,all_emb=False,crf=True,dropout=0.5,lr_method=sgd-lr_.002,prefix=0,external_features=pos.1.3chunk.2.3wh.3.3if.4.3s.5.3",
    #     1: "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-scripts/script-1b-emb-crf-f12345-lr002/models/tag_scheme=iobes,word_dim=50,word_bidirect=True,pre_emb=rre.w2v50.txt,all_emb=False,crf=True,dropout=0.5,lr_method=sgd-lr_.002,prefix=0,external_features=pos.1.3chunk.2.3wh.3.3if.4.3s.5.3layer1.7.5",
    #     2: "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-scripts/script-1c-emb-crf-f12345-lr002/models/tag_scheme=iobes,word_dim=50,word_bidirect=True,pre_emb=rre.w2v50.txt,all_emb=False,crf=True,dropout=0.5,lr_method=sgd-lr_.002,prefix=0,external_features=pos.1.3chunk.2.3wh.3.3if.4.3s.5.3layer1.7.5layer2.8.5"
    # }

    # input = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/test-3layer/fold.0.test.conll"

    input = input_folder + "/" + file
    lines = common.get_all_lines(input)
    newlines = []
    for line in lines:
        if line.strip() != "":
            newline = "\t".join(line.strip().split("\t")[:-nlayer] + ["O"]) # remove all gold label + append O tag
            newlines.append(newline)
        else:
            newlines.append("")

    input_files = [folder + "/" + file + ".temp.layer" + str(i) for i in range(nlayer + 1)]
    eval_files = [folder + "/" + file + ".eval" + str(i) for i in range(nlayer)]
    result_files = [folder + "/" + file + ".result.layer" + str(i) + ".txt" for i in range(nlayer)]

    save_all_lines(newlines, input_files[0]) # create input file of layer 0

    eval_lines_all = []
    for i in range(nlayer):

        parameters = [
            "--test_file " + input_files[i],
            "--out_file " + input_files[i + 1],
            "--model " + models[i],
            "--prefix " + "ABC"
        ]

        os.system("python predict_file.py " + ' '.join(parameters))
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



# def evaluate2(test_file, models):
#     models = {
#         0: "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-scripts/script-1a-emb-crf-f12345-lr002/models/tag_scheme=iobes,word_dim=50,word_bidirect=True,pre_emb=rre.w2v50.txt,all_emb=False,crf=True,dropout=0.5,lr_method=sgd-lr_.002,prefix=0,external_features=pos.1.3chunk.2.3wh.3.3if.4.3s.5.3",
#         1: "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-scripts/script-1b-emb-crf-f12345-lr002/models/tag_scheme=iobes,word_dim=50,word_bidirect=True,pre_emb=rre.w2v50.txt,all_emb=False,crf=True,dropout=0.5,lr_method=sgd-lr_.002,prefix=0,external_features=pos.1.3chunk.2.3wh.3.3if.4.3s.5.3layer1.7.5",
#         2: "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-scripts/script-1c-emb-crf-f12345-lr002/models/tag_scheme=iobes,word_dim=50,word_bidirect=True,pre_emb=rre.w2v50.txt,all_emb=False,crf=True,dropout=0.5,lr_method=sgd-lr_.002,prefix=0,external_features=pos.1.3chunk.2.3wh.3.3if.4.3s.5.3layer1.7.5layer2.8.5"
#     }
#
#     input = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/test-3layer/fold.0.test.conll"
#
#     lines = common.get_all_lines(input)
#     newlines = []
#     for line in lines:
#         if line.strip() != "":
#             newline = "\t".join(line.strip().split("\t")[:-3] + ["O"])
#             newlines.append(newline)
#         else:
#             newlines.append("")
#
#     folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/test-3layer/temp"
#     file = "fold.0.test.conll"
#
#     nlayer = 3
#     input_files = [folder + "/" + file + ".temp.layer" + str(i) for i in range(nlayer + 1)]
#     eval_files = [folder + "/" + file + ".eval" + str(i) for i in range(nlayer)]
#     result_files = [folder + "/" + file + ".result.layer" + str(i) + ".txt" for i in range(nlayer)]
#
#     save_all_lines(newlines, input_files[0])
#
#     eval_lines_all = []
#     for i in range(nlayer):
#
#         parameters = [
#             "--test_file " + input_files[i],
#             "--out_file " + input_files[i + 1],
#             "--model " + models[i],
#             "--prefix " + "ABC"
#         ]
#
#         os.system("python predict_file.py " + ' '.join(parameters))
#         lines_i = common.get_all_lines(input_files[i + 1])
#         print "#LEN: ", len(lines_i)
#
#         # output conll to conlleval
#         lines_eval = []
#         for k in range(len(lines)):
#             if lines[k].strip() != "":
#                 tokens1 = lines[k].strip().split("\t")
#                 tokens2 = lines_i[k].strip().split("\t")
#                 assert zero_digits(tokens1[0]) == zero_digits(tokens2[0])
#                 lines_eval.append(" ".join([tokens1[0], tokens1[i - nlayer], tokens2[-2]]))
#             else:
#                 lines_eval.append("")
#
#         save_all_lines(lines_eval, eval_files[i])
#         eval_lines = call_conlleval(eval_files[i], result_files[i])
#         eval_lines_all.append("=======================")
#         eval_lines_all.append("LAYER: " + str(i))
#         eval_lines_all.extend(eval_lines)
#
#     for l in eval_lines_all:
#         print l

def evaluate_new(model_folder, result_file_path, test_folder, temp, nfold, nlayer):
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

        models = {
            0: model_folder + "/layer0/models/prefix=%d" % (i),
            1: model_folder + "/layer1/models/prefix=%d" % (i),
            # 2: model_folder + "/layer2/models/prefix=%d" % (i)
        }
        assert os.path.isfile(test_folder + "/" + test_file)
        assert os.path.isdir(models[0])
        assert os.path.isdir(models[1])
        # assert os.path.isdir(models[2])
        result_file.write("\n-------------------------------------------------------------------------------------\n")
        result_file.write("FOLD %i :\n" % (i))
        result_file.write("test file: " + test_folder + "/" + test_file + "\n")
        result_file.write("model layer 0: " + models[0] + "\n")
        result_file.write("model layer 1: " + models[1] + "\n")
        # result_file.write("model layer 2: " + models[2] + "\n")
        evaluate(test_folder, test_file, models, temp, result_file, nlayer)

if __name__ == '__main__':
    model_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/run-separate-layer/01emb-crf-d100"
    result_file_path = model_folder + "/01emb-crf-d100.txt"
    test_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/data/conll-2layer"
    temp = model_folder + "/out-temp"
    nfold = 1
    nlayer = 2

    # model_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-v1/01jpc-emb-crf-d100/00emb-crf-nofeature-lr002"
    # result_file_path = model_folder + "/00emb-crf-nofeature-lr002.txt"
    # test_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/test-3layer"
    # temp = model_folder + "/out-temp"
    # nfold = 1

    evaluate_new(model_folder, result_file_path, test_folder, temp, nfold, nlayer)