import pickle
import codecs
import json
from scipy import spatial
from collections import Counter

def loadJSONData(dataPath):
    dataJSONText = codecs.open(dataPath, encoding='utf8')
    return json.load(dataJSONText)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def save_object_to_json(obj, filename):
    with open(filename, 'w') as fp:
        json.dump(obj, fp)

def get_all_lines(file):
    with open(file) as f:
        return f.readlines()

