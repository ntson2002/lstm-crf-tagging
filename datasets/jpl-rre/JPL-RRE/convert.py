import sys
import codecs
reload(sys)
sys.setdefaultencoding("utf-8")

folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data-japanese/train-test"
folderout = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data-japanese/train-test2"
files = ["test." + str(i) + ".txt" for i in range(1,11)] + ["train." + str(i) + ".txt" for i in range(1,11)]

def get_all_lines(file):
    with open(file) as f:
        return f.readlines()

def convert(filein, fileout):
	lines = get_all_lines(filein)
	lines2 = []
	for line in lines:
		lines2.append("\t".join(line.split()))
	with codecs.open(fileout, "w", encoding="utf-8") as f:
		f.write('\n'.join(lines2).encode("UTF-8"))

for fname in files:
	filein = folder + "/" + fname
	fileout = folderout + "/" + fname
	convert(filein, fileout)