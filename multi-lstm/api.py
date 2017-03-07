import web
import json
import os
import time
import codecs
import numpy as np
import loader

from utils import predict_multilayer
from loader import prepare_dataset3
from loader import update_tag_scheme
from model import Model


urls = (
        # '/api/tagging/text=(.*)', 'api_do_tagging',
        '/api/tagging', 'api_do_tagging',
        '/demo/(.*)', 'demo')

# loading the model
model_path = os.getenv("HOME") + "/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/run-multi-lstm/models/tag_scheme=iobes,char_dim=25,word_dim=100,word_bidirect=True,pre_emb=train.txt.w2vec100,all_emb=False,crf=True,dropout=0.5,external_features=pos.1.30chunk.2.30,prefix=p30c30"


# Initialize model
def load_model(model_path):
    model = Model(model_path=model_path)
    parameters = model.parameters

    # Data parameters
    lower = parameters['lower']
    zeros = parameters['zeros']
    tag_scheme = model.parameters['tag_scheme']

    # Load reverse mappings
    word_to_id, char_to_id = [
        {v: k for k, v in x.items()}
        for x in [model.id_to_word, model.id_to_char]
    ]

    print 'Reloading previous model...'
    _, f_eval = model.build(training=False, **parameters)
    model.reload()
    return [f_eval, model, parameters, lower, zeros, tag_scheme, word_to_id, char_to_id]

f_eval, model, parameters, lower, zeros, tag_scheme, word_to_id, char_to_id = load_model(model_path)

class demo:
    def GET(self, q):
        import codecs
        web.header('Content-Type', 'text/html')
        with codecs.open("demo.html", "r", "utf-8") as f:
            html_text = f.read()
        return html_text

class api_do_tagging:
    def POST(self):
        web.header('Content-Type', 'application/json')
        post_data = web.input(_method='post')
        # print post_data


        # read the data from request and save into file
        file_id = np.random.randint(1000000, 2000000)
        input_path = os.path.join("tmp", "user.%i.input" % file_id)
        with codecs.open(input_path, "w", "utf-8") as f:
            f.write(post_data["text"])

        test_sentences = loader.load_sentences(input_path, lower, zeros)
        update_tag_scheme(test_sentences, tag_scheme)

        test_data = prepare_dataset3(
            test_sentences, word_to_id, char_to_id, model.tag_maps, model.feature_maps, lower
        )

        # print test_data[0]

        out_sentences = predict_multilayer(parameters, f_eval, test_sentences, test_data, model.tag_maps, None)

        text = ""
        # predictions_list = [p.split("\t") for p in predictions]
        text = " ".join([line[0] for s in out_sentences for line in s])

        data = {"sentences": out_sentences, "text": text}

        return json.dumps(data, indent=4, sort_keys=True, encoding="utf-8")


class TaggerAPIApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))


if __name__ == "__main__":
    app = TaggerAPIApplication(urls, globals())
    app.run(port=8124)


