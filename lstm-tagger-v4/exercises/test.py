from glob import glob
from pprint import pprint

parent = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/scripts-3/models/"
paths = glob(parent+"*/")
pprint (paths)
