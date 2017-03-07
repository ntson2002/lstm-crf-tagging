__author__ = 'sonnguyen'
import os
import codecs


def convert_folder_to_json(input_folder, output_folder, column=5):
    for file in os.listdir(input_folder):
        if file.endswith(".txt"):
            output_file = output_folder + "/" + file.replace(".txt", ".conll")
            convert_org_file_to_conll(input_folder + "/" + file, output_file)


def convert_org_file_to_conll(input_file, output_file):
    f = codecs.open(input_file, 'r', 'UTF-8')
    file_out = codecs.open(output_file, "w", "utf8")
    start = False
    sentences = []
    num_token = 0
    for line in f:
        line = line.strip()
        if "<s>" not in line:
            if "</s>" not in line:
                # print line + "\tO"
                file_out.write(line + "\tO\n")
            else:
                # print
                file_out.write("\n")


def main_convert_test_data():

    input_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/data/NER2016-TestData/Test-column"
    output_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output-test/conll-format"
    # input_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/data/test"
    convert_folder_to_json(input_folder, output_folder)

if __name__ == '__main__':
    main_convert_test_data()
