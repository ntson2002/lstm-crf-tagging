__author__ = 'sonnguyen'
import os
import codecs
import common
from sklearn.cross_validation import train_test_split
import numpy as np
from math import ceil

def convert_folder_to_json(input_folder, output_folder, column=5):
    for file in os.listdir(input_folder):
        if file.endswith(".txt"):
            output_file = output_folder + "/" + file.replace(".txt", ".json")
            convert_file_to_json(input_folder + "/" + file, output_file, column)

def convert_file_to_json(input_file, output_file, column=5):
    f = codecs.open(input_file, 'r', 'UTF-8')
    start = False
    sentences = []
    num_token = 0
    a_sentence = []

    i = 1
    for line in f:
        i = i + 1
        line = line.strip()
        if not start and "-DOCSTART-" in line:
            start = True
            continue


        if start:
            if "<s>" in line:
                a_sentence = []
                error = False
            elif "</s>" in line:
                if not error:
                    sentences.append(a_sentence)
            else:
                if line != "":
                    items = line.split("\t")
                    num_token += 1

                    if len(items) == column:
                        a_sentence.append(items)

                    if len(items) == column and items[3] not in {"O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-TIME", "I-TIME", "B-MISC", "I-MISC"}:
                        print "error2", i, input_file, line
                        error = True

                    if len(items) == column and len([s for s in items if s.strip() == ""]) > 0:
                        print "error2", i, input_file, line
                        error = True

                    if len(items) != column:
                        print "error3", i, input_file, line
                        a_sentence.append(items)
                        error = True

                    if len(items) == column and len([s for s in items if " " in s.strip()]) > 0:
                        print "error4", i, input_file, line
                        a_sentence.append(items)
                        error = True

    if not start:
        print input_file, "have no -DOCSTART-"

    # print input_file + "\t" + str(len(sentences)) + "\t" + str(num_token)
    common.save_object_to_json(sentences, output_file)


def create_conll_format_from_json(input_json_folder,
                                  split_type,
                                  output_file_all=None,
                                  output_folder = None,
                                  percent=None,
                                  nfold=None,
                                  twolayer=False):

    def write_a_batch(sentences, path, twolayer=False):
        file_out = codecs.open(path, "w", "utf8")
        for sentence in sentences:
            for tokens in sentence:
                # print tokens
                # if str(type(tokens)) == "<type 'NoneType'>":
                #     print sentence
                label = tokens[3]
                if label not in {"B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-TIME", "I-TIME", "B-MISC", "I-MISC"}:
                    label = "O"

                if twolayer:
                    ss = [tokens[0], tokens[1], tokens[2], label, tokens[4]]
                else:
                    ss = [tokens[0], tokens[1], tokens[2], label]

                file_out.write("\t".join(ss) + "\n")

            file_out.write("\n")

    if "merge" in split_type:
        all_sentences = []
        for file in os.listdir(input_json_folder):
            if file.endswith(".json"):
                sentences = common.loadJSONData(input_json_folder + "/" + file)
                all_sentences.extend(sentences)

        print "total of sentences: ", len(all_sentences)


    if split_type == "merge_all":
        write_a_batch(all_sentences, output_file_all, twolayer)

    if split_type == "merge_and_split_by_percent":
        all_ids = range(len(all_sentences))
        train_ids, test_ids = train_test_split(all_ids, test_size=percent, random_state=42)
        train_sentences = [all_sentences[i] for i in train_ids]
        test_sentences = [all_sentences[i] for i in test_ids]
        write_a_batch(train_sentences, output_folder + "/train.conll", twolayer)
        write_a_batch(test_sentences, output_folder + "/test.conll", twolayer)

    if split_type == "merge_and_split_cv":
        np.random.seed(42)
        all_ids = np.random.permutation(range(len(all_sentences)))
        folds = [(int(ceil(len(all_ids) / float(nfold) * i)), int(ceil(len(all_ids) / float(nfold) * (i+1)) - 1)) for i in range(nfold)]

        for i in range(nfold):
            print folds[i]
            test_ids_i = [all_ids[k] for k in range(folds[i][0], folds[i][1]+1)]
            train_ids_i = [all_ids[k] for k in range(len(all_ids)) if k < folds[i][0] or k > folds[i][1]]
            print len(test_ids_i), len(train_ids_i), len(test_ids_i) + len(train_ids_i)
            train_sentences_i = [all_sentences[k] for k in train_ids_i]
            test_sentences_i = [all_sentences[k] for k in test_ids_i]
            write_a_batch(train_sentences_i, output_folder + "/" + str(i) + "_train.conll", twolayer)
            write_a_batch(test_sentences_i, output_folder + "/" + str(i) + "_test.conll", twolayer)

    if split_type == "separated_file":
        for file in os.listdir(input_json_folder):
            if file.endswith(".json"):
                sentences = common.loadJSONData(input_json_folder + "/" + file)
                write_a_batch(sentences, output_folder + "/" + file.replace(".json", ".conll"), twolayer)

def extract_ner_from_conll_format(path, output):
    def extract_ner(sentence, path):
        i = 0
        ners = []
        while i < len(sentence):
            if sentence[i][3][0] == "B":
                label = sentence[i][3][2:]
                aner = sentence[i][0]
                previous = ""
                if i > 0:
                    previous = sentence[i-1][0]

                i = i + 1
                while i < len(sentence) and sentence[i][3][0] == "I":
                    aner = aner + " " + sentence[i][0]
                    i = i + 1
                # print label + "\t" + aner


                ners.append([label, aner, previous])
                print label, "\t", aner, "\t", previous
            else:
                if sentence[i][3][0] == "I":
                    print "#error", path, "\t".join(sentence[i])
                i = i + 1
        return ners
        # ners = [r[0] + "/" + r[3] for r in sentence if r[3][0] in {"B", "I"}]
        # if len(ners) > 0:
        #     print " ".join(ners)

    f = codecs.open(path, 'r', 'UTF-8')
    sentences = []
    a_sentence = []
    for line in f:
        line = line.strip()
        if line == "":
            if len(a_sentence) > 0:
                sentences.append(a_sentence)
            a_sentence = []
        else:
            items = line.split("\t")
            a_sentence.append(items)
    all_ners = []
    for sentence in sentences:
        ners = extract_ner(sentence, path)
        all_ners.extend(ners)
    # print all_ners
    common.save_object_to_json(all_ners, output)
    # print len(sentences[0])
    # print len(all_ners)
    # print sentences[0]

def check_conll_format(path):
    f = codecs.open(path, 'r', 'UTF-8')
    for line in f:
        line = line.strip()
        if line != "":
            items = line.split()
            if len(items) != 5:
                print "error", line



def main_convert_train_data():
    # function = "extract_ner_from_conll_format"
    # function = "convert_folder_to_json"
    # function = "create_conll_format_from_json"
    # function = "extract_ner_from_conll_format_folder"
    function = "check_conll_format"

    if function == "convert_folder_to_json":
        # train-data
        # input_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/data/NER2016-training_data-1-fixerror/testa"
        # output_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/data/NER2016-training_data-1-json/testa"

        input_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/data/Reference-TestData/NER2016-Testdata-Gold"
        output_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/data/NER2016-training_data-1-json/testb"
        convert_folder_to_json(input_folder, output_folder)


    elif function == "create_conll_format_from_json":
        # input_json_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/json"
        # # create 1 file
        # output_file_all = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/all/all_data_ner.conll"
        # create_conll_format_from_json(input_json_folder, split_type="merge_all", output_file_all=output_file_all)
        #
        # # split into train and test by percent
        # output_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/split"
        # create_conll_format_from_json(input_json_folder, split_type="merge_and_split_by_percent", output_folder=output_folder, percent=0.33)
        #
        # # split into folds for cross validation
        # output_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/cv"
        # create_conll_format_from_json(input_json_folder, split_type="merge_and_split_cv", output_folder=output_folder, nfold=10)
        #
        # # convert each input file seperate
        # output_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/conll"
        # create_conll_format_from_json(input_json_folder, split_type="separated_file", output_folder=output_folder)


        # input_json_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/json/train"
        # output_file_all = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/split2/train.conll"
        # create_conll_format_from_json(input_json_folder, split_type="merge_all", output_file_all=output_file_all)
        #
        # input_json_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/json/test"
        # output_file_all = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/split2/test.conll"
        # create_conll_format_from_json(input_json_folder, split_type="merge_all", output_file_all=output_file_all)

        # input_json_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/json-test"
        # output_file_all = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/all-test/ner-test.conll"
        # create_conll_format_from_json(input_json_folder, split_type="merge_all", output_file_all=output_file_all)

        input_json_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/data/NER2016-training_data-1-json/testb"
        output_file_all = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/data/NER2016-training_data-1-json/testb.conll"
        create_conll_format_from_json(input_json_folder, split_type="merge_all", output_file_all=output_file_all, twolayer=False)


    elif function == "extract_ner_from_conll_format":
        path = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/cv/0_train.conll"
        out_path = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/all_ner.json"
        extract_ner_from_conll_format(path, out_path)

    elif function == "extract_ner_from_conll_format_folder":
        folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output/conll"
        for file in os.listdir(folder):
            if file.endswith(".conll"):
                # print file
                extract_ner_from_conll_format(folder + "/" + file)

    elif function == "check_conll_format":
        path = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/data/NER2016-training_data-1-json/conll-2layer/train2.conll"
        # path = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/experiments/data-chien-crf/train.conll"
        # path = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/experiments/evaluation/temp/eval.1262207.output"
        check_conll_format(path)


if __name__ == '__main__':
    main_convert_train_data()
