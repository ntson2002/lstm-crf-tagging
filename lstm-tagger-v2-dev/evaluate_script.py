import os

test_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/en-ner/scripts-4-en/results/test"
models_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/en-ner/scripts-7/models"
out_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/en-ner/scripts-7/results"

folders = [name for name in os.listdir(models_folder)]

print folders
for folder in folders:
    items = folder.split(",")
    table = {}
    for item in items:
        ss = item.split("=")
        # print ss
        table[ss[0]] = ss[1]

    parameters = [
        "--test_folder " + test_folder,
        "--out_folder " + out_folder,
        "--model " + models_folder + "/" + folder,
        "--prefix " + "p" + table["pos_dim"] + "_c" + table["chunk_dim"] + "_w" + table["word_dim"] + "_emb_"+table["pre_emb"] + "_",
        "--eval_script conlleval2",
        "--show_detail 1"
    ]

    os.system("python evaluate.py " + ' '.join(parameters))