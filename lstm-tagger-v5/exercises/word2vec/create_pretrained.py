import loader
import optparse
from collections import OrderedDict
from loader import word_mapping
import gensim
import pickle
import codecs


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)

# Read parameters from command line


# word2vec_file = "/work/vietld/mikolov/onebill256.bin"
# binary=True
# train = "/home/s1520203/programs/sentence-compression-en/data/vietdata/all.conll"
# out_pickle = "/home/s1520203/programs/sentence-compression-en/data/vietdata/all.en.sc.w2vec256"
# out_txt = "/home/s1520203/programs/sentence-compression-en/data/vietdata/all.en.sc.txt.w2vec256"

# Default value
# word2vec_file = "/home/s1520203/programs/word2vec/output/wiki-data-tokenized.dat-100.bin"
# binary=True
# train = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/dev/train.conll"
# out_pickle = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/pre-trained/train.pickle.wiki.w2vec100"
# out_txt = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/vn-ner/data/pre-trained/train.txt.wiki.w2vec100"

# word2vec_file = "/home/s1520203/GoogleNews-vectors-negative300.bin"
# train = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v2-dev/data/ner-en/dev/eng.train.iob2"
# out_pickle = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v2-dev/data/ner-en/pre-trained/eng.train.pickle.w2vec300"
# out_txt = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v2-dev/data/ner-en/pre-trained/eng.train.txt.w2vec300"


# word2vec_file = "/work/sonnguyen/glove/glove_word2vec/glove.twitter.27B.100d.w2vec"
# out_txt = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v2-dev/data/ner-en/pre-trained/train-gt27b100.txt"
# word2vec_file = "/work/sonnguyen/glove/glove_word2vec/glove.6B.100d.w2vec"
# binary=False
# train = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v2-dev/data/ner-en/dev/eng.all"
# out_pickle = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v2-dev/data/ner-en/pre-trained/eng.train.pickle.w2vec100"
# out_txt = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v2-dev/data/ner-en/pre-trained/train-g6b100.txt"


# word2vec_file = "/home/s1520203/programs/word2vec/output/wiki-data-tokenized.dat-100.bin"
# binary=True
# train = "/home/s1520203/programs/sentence-compression-vi/data/tokenize/gold.vnexpress.corpus.conll"
# out_pickle = "/home/s1520203/programs/sentence-compression-vi/data/tokenize/vnexpress/pretrained/gold.vnex.w2vec100"
# out_txt = "/home/s1520203/programs/sentence-compression-vi/data/tokenize/vnexpress/pretrained/gold.vnex.w2vec100.txt"


# word2vec_file = "/home/s1520203/programs/word2vec/output/wiki-data-tokenized.dat-100.bin"
# binary=True
# train = "/home/s1520203/programs/sentence-compression-vi/data/tokenize/first-data/vn.unicode.tok.conll"
# out_pickle = "/home/s1520203/programs/sentence-compression-vi/data/tokenize/first-data/pretrained/vn.unicode.w2v100"
# out_txt = "/home/s1520203/programs/sentence-compression-vi/data/tokenize/first-data/pretrained/vn.unicode.w2v100.txt"


# word2vec_file = "/home/s1520203/programs/word2vec/output/japan-law.en-50.bin"
# binary=True
# train = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/_out.rand.feature_layer1.conll"
# out_pickle = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/pretrained/rre.w2v50"
# out_txt = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/pretrained/rre.w2v50.txt"


word2vec_file = "/home/s1520203/programs/word2vec/output/japan-law.en-100.bin"
binary=True
train = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/new2.conll"
out_pickle = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/pretrained/rre.w2v100"
out_txt = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/pretrained/rre.w2v100.txt"


optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default=train,
    help="Train set location"
)

optparser.add_option(
    "-l", "--lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)

optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)

optparser.add_option(
    "-A", "--word2vec_file", default=word2vec_file,
    help="word2vec file"
)

optparser.add_option(
    "-B", "--out_pickle", default=out_pickle,
    help="output file that contains dictionary (pickle) object of all extracted pretrained vector"
)

optparser.add_option(
    "-C", "--out_txt", default=out_txt,
    help="output file that contains all extracted pretrained vector in txt format (word vector)"
)

opts = optparser.parse_args()[0]

parameters = OrderedDict()
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1

lower = parameters['lower']
zeros = parameters['zeros']

print "loading word2vec file ...",
# model = gensim.models.Word2Vec.load_word2vec_format(opts.word2vec_file, binary=True, unicode_errors='ignore')
model = gensim.models.Word2Vec.load_word2vec_format(opts.word2vec_file, binary=binary, unicode_errors='ignore')

print "finish"

train_sentences = loader.load_sentences(opts.train, lower, zeros)
dico_words_train = word_mapping(train_sentences, lower)[0]

count_yes = 0
count_no = 0
dic = {}
not_found_words = []
for w, c in dico_words_train.items():
    if w in model:
        dic[w] = model[w]
        count_yes += 1
    else:
        count_no += 1
        not_found_words.append(w)
print

print "#found: ", count_yes, "; not found: ", count_no
print "#top 100 word not found", ' '.join(not_found_words[:100])
print "#pre trained vectors: ", len(dic)
if (opts.out_pickle != ""):
    print "saving to pickle file ... ",
    save_object(dic, opts.out_pickle)
    print "finish. Location:", opts.out_pickle

if (opts.out_txt != ""):
    print "saving to pickle file ... ",
    with codecs.open(opts.out_txt, "w", "utf-8") as f:
        for w, v in dic.items():
            f.write(w + " " + " ".join(map(str, v)) + "\n")
        print "finish. Location:", opts.out_txt

# print train_sentences[0]
# print len(train_sentences)
# for s in train_sentences:
#     a = [t[0] for t in s]
#     print ' '.join(a)
#     # print s[0][0]


