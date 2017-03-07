import codecs

input_file = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/ger-ner/tsv/NER-de-test.tsv"
output_file = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/ger-ner/data/layer1/NER-de-test-layer2.conll"
f = codecs.open(input_file, 'r', 'UTF-8')
f_out = codecs.open(output_file, 'w', 'UTF-8')

for line in f:
    line = line.strip()
    if not line.startswith("#"):
        items = line.split("\t")
        # print len(items)
        if len(items) != 4 and len(items) != 1:
            print line, len(items)
        if len(items) == 1:
            f_out.write("\n")
        else:
            x = [items[1], 'O', items[2], items[3]]
            f_out.write("\t".join(x) + "\n")

f.close()
f_out.close()

