import web
import json
import os
import time
import codecs
import optparse
import numpy as np
from loader import prepare_sentence
from utils import create_input2, iobes_iob, zero_digits
from model import Model

urls = ('/api/dotagging/(.*)', 'api_do_tagging',
        '/api/f=topics', 'api_topics',
        '/api/search2/(.*)/topic/(.+)', 'api_query_topic')

 # loading the model
model_path = "/Volumes/s1520203/programs/vn-ner-tagger/models/vn-ner"
model = Model(model_path=model_path)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval = model.build(training=False, **parameters)
model.reload()


class api_topics:
    def GET(self, q):
        web.header('Content-Type', 'application/json')
        # data = api.getTopics()
        data = ["a", "b", "c", "d", q]
        return json.dumps(data, indent=4, sort_keys=True)

class api_do_tagging:
    def POST(self, username):
        post_input = web.input(_method='post')
        web.header('Content-Type', 'application/json')
        data = post_input
        # print data

        f_output = codecs.open("output2.txt", 'w', 'utf-8')
        start = time.time()
        delimiter = "__"

        print 'Tagging...'
        # with codecs.open("input.txt", 'r', 'utf-8') as f_input:
        count = 0
        sentences = data["text"].split("\n")
        results = []
        sentences2 = [sentence for sentence in sentences if sentence.strip() != ""]
        for line in sentences2:
            words = line.rstrip().split()
            if line:
                # Lowercase sentence
                if parameters['lower']:
                    line = line.lower()
                # Replace all digits with zeros
                if parameters['zeros']:
                    line = zero_digits(line)
                # Prepare input
                sentence = prepare_sentence(words, word_to_id, char_to_id,
                                            lower=parameters['lower'])
                input = create_input2(sentence, parameters, False, False)
                # Decoding
                if parameters['crf']:
                    y_preds = np.array(f_eval(*input))[1:-1]
                else:
                    y_preds = f_eval(*input).argmax(axis=1)
                y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
                # Output tags in the IOB2 format
                if parameters['tag_scheme'] == 'iobes':
                    y_preds = iobes_iob(y_preds)
                # Write tags
                assert len(y_preds) == len(words)
                # result = '%s\n' % ' '.join('%s%s%s' % (w, delimiter, y) for w, y in zip(words, y_preds))
                result = [ [w, y] for w, y in zip(words, y_preds)]
                results.append(result)
                f_output.write('%s\n' % ' '.join('%s%s%s' % (w, delimiter, y)
                                                 for w, y in zip(words, y_preds)))
            else:
                f_output.write('\n')
            count += 1
            if count % 100 == 0:
                print count

        print '---- %i lines tagged in %.4fs ----' % (count, time.time() - start)
        f_output.close()
        data["sentences"] = results
        return json.dumps(data, indent=4, sort_keys=True, encoding="utf-8")


class TaggerAPIApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

if __name__ == "__main__":
    app = TaggerAPIApplication(urls, globals())
    app.run(port=8081)


