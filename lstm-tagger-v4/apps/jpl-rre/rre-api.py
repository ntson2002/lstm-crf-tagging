#!/usr/bin/python
# -*- coding: utf-8 -*-
import web
import json
import os
from loader import update_tag_scheme, prepare_dataset2
from utils import predict2
from model import Model
import CaboCha

PUNC_LIST = {u'、', u"。"}

c = CaboCha.Parser()
def convert_sentence(sentence):
    global c
    tree = c.parse(sentence)
    data = []
    chunks = []
    for i in range(tree.chunk_size()):
        chunk = tree.chunk(i)
        p = []
        for ix in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
            p.append(tree.token(ix).surface)
        last_word = p[-1].decode("utf-8")
        chunks.append("".join(p).decode("utf-8"))

        if last_word in PUNC_LIST:
            pw = last_word
            p = p[:-1]
        else:
            pw = u'NO'

        hw = p[chunk.head_pos].decode("utf-8")
        fw = p[chunk.func_pos].decode("utf-8")
        data.append([hw, u'0000', u'000000', fw, u'0000', u'000000', pw, u'O'])

    return data, chunks

urls = ('/api/parse/', 'RreParse',
        '/demo/(.*)', 'Demo',
        '/api/test/(.*)', 'Test')

"""
    Load the model
"""

if True:
    model_path = os.getenv("HOME") + "/Bitbucket/lstm-crf-tagging/lstm-tagger-v4/apps/jpl-rre/rre-sample-scripts/models/jpl-rre-f36"
    model = Model(model_path=model_path)
    parameters = model.parameters

    # Data parameters
    lower = parameters['lower']
    zeros = parameters['zeros']
    tag_scheme = model.parameters['tag_scheme']

    # Load reverse mappings
    word_to_id, char_to_id, tag_to_id = [
        {v: k for k, v in x.items()}
        for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
        ]

    print  model.id_to_tag
    print 'Reloading previous model...'
    _, f_eval = model.build(training=False, **parameters)
    model.reload()

    print 'Loading ... finish !!!'


class RreParse:
    def POST(self):
        def btag(token):
            return token[1][0]

        def logic_tag(token):
            return "O" if token[1] == "O" else token[1][2:]

        def word(token):
            return token[0]


        post_input = web.input(_method='post')
        sentences = post_input["text"].strip().split("\n")
        print "receive: \n", "\n".join(sentences)
        web.header('Content-Type', 'application/json')
        test_sentences = []
        chunks_sentences = []

        for s in sentences:
            conll, chunk_s = convert_sentence(s.strip().encode("utf-8"))
            test_sentences.append(conll)
            chunks_sentences.append(chunk_s)

        update_tag_scheme(test_sentences, tag_scheme)

        test_data = prepare_dataset2(
            test_sentences, word_to_id, char_to_id, tag_to_id, model.feature_maps, lower
        )

        predictions_sentences = predict2(parameters, f_eval, test_sentences, test_data, model.id_to_tag)

        # re_parts = [ [item[0][0], item[1][-1]] for i in range(len(sentences)) for item in zip(chunks_sentences[i], predictions_sentences[i])]
        re_parts_iob_sentences = []
        re_parts_sentences = []
        for i in range(len(sentences)):
            tokens = [[item[0], item[1][-1]] for item in zip(chunks_sentences[i], predictions_sentences[i])]
            re_parts_iob_sentences.append(tokens)
            # chunks = chunks_sentences[i]
            # predictions = predictions_sentences[i]

            for token in tokens:
                print token[0], token[1]

            re_parts = []
            j = 0
            while (j < len(tokens)):
                tag = logic_tag(tokens[j])
                if btag(tokens[j]) == "B":

                    part = [word(tokens[j])]
                    j += 1
                    while j < len(tokens) and btag(tokens[j]) in {"I", "E"}:
                        part.append(word(tokens[j]))
                        j += 1
                    re_parts.append([part, tag])

                elif btag(tokens[j]) == "O":
                    part = [word(tokens[j])]
                    j += 1
                    while j < len(tokens) and btag(tokens[j]) == "O":
                        part.append(word(tokens[j]))
                        j += 1
                    re_parts.append([part, tag])
                else:
                    j += 1

            re_parts_sentences.append(re_parts)

        print "finish !!!"


        data = {
            "sentences": sentences,
            "chunks": chunks_sentences,
            "re_parts_iob" : re_parts_iob_sentences,
            "re_parts": re_parts_sentences
        }
        return json.dumps(data, indent=4, sort_keys=True)


class Demo:
    def GET(self, q):
        import codecs
        web.header('Content-Type', 'text/html')        
        html_text = "<h1>none</h1>"

        with codecs.open("demo.html", "r", "utf-8") as f:
            html_text = f.read()
        return html_text


class Test:
    def GET(self, q):
        web.header('Content-Type', 'application/json')
        # data = api.getTopics()
        data = ["a", "b", "c", "d", q]
        return json.dumps(data, indent=4, sort_keys=True)


class TaggerAPIApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

if __name__ == "__main__":
    app = TaggerAPIApplication(urls, globals())
    app.run(port=8126)


