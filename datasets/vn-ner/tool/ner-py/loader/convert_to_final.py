__author__ = 'sonnguyen'
import os
import codecs


def convert_folder_to_final_output(input_folder, output_folder, column=5):
    for file in os.listdir(input_folder):
        if file.endswith(".conll"):
            output_file = output_folder + "/" + file.replace(".conll", ".txt")
            convert_file_to_final_output(input_folder + "/" + file, output_file)


def convert_file_to_final_output(input_file, output_file):
    f = codecs.open(input_file, 'r', 'UTF-8')
    file_out = codecs.open(output_file, "w", "utf8")

    all_sentences = []
    a_sentence = []
    for line in f:
        line = line.strip()
        # print line
        if line == "":
            if (len(a_sentence) > 0):
                all_sentences.append(a_sentence)
            a_sentence = []
        else:
            items = line.split("\t")
            new_items  = items[:len(items)-2]
            new_items.extend([items[-1]])

            a_sentence.append("\t".join(new_items))

    if (len(a_sentence) != 0):
        all_sentences.append(a_sentence)


    print len(all_sentences)


    for a_sentence in all_sentences:
        file_out.write("<s>\n" + "\n".join(a_sentence) + "\n</s>\n")


def main_convert_to_final():

    input_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output-test/output-crf"
    output_folder = "/Users/sonnguyen/Bitbucket/vlsp2016-ner/ner-py/output-test/jaist-ner-system3"

    convert_folder_to_final_output(input_folder, output_folder)

if __name__ == '__main__':
    main_convert_to_final()
