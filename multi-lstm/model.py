import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import cPickle

from utils import shared, set_values, get_name, get_path
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward, forward1
from optimization import Optimization


class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)

            # Model location
            model_path = os.path.join(models_path, get_path(parameters))

            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                self.parameters = cPickle.dump(parameters, f)
        else:
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()
        self.components = {}



    # def save_mappings(self, id_to_word, id_to_char, id_to_tag, feature_maps):
    #     """
    #     We need to save the mappings if we want to use the model later.
    #     """
    #     self.id_to_word = id_to_word
    #     self.id_to_char = id_to_char
    #     self.id_to_tag = id_to_tag
    #     self.feature_maps = feature_maps
    #     # self.id_to_pos = id_to_pos
    #     # self.id_to_chunk = id_to_chunk
    #
    #     with open(self.mappings_path, 'wb') as f:
    #         mappings = {
    #             'id_to_word': self.id_to_word,
    #             'id_to_char': self.id_to_char,
    #             'id_to_tag': self.id_to_tag
    #         }
    #         cPickle.dump([feature_maps, mappings], f)

    def save_mappings2(self, id_to_word, id_to_char, tag_maps, feature_maps):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        # self.id_to_tag = id_to_tag
        self.feature_maps = feature_maps
        self.tag_maps = tag_maps
        # self.id_to_pos = id_to_pos
        # self.id_to_chunk = id_to_chunk

        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char
                # 'id_to_tag': self.id_to_tag
            }
            cPickle.dump([feature_maps, tag_maps, mappings], f)


    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            [feature_maps, tag_maps, mappings] = cPickle.load(f)

        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']

        self.feature_maps = feature_maps
        self.tag_maps = tag_maps

        print self.tag_maps

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              char_bidirect,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              pre_emb,
              crf,
              cap_dim,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters

        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)

        n_tags = len(self.tag_maps[0]['id_to_tag'])

        if len (self.tag_maps) > 1:
            n_tags1 = len(self.tag_maps[1]['id_to_tag'])
            print "n_tags1: ", n_tags1

        # n_poss = len(self.id_to_pos)
        # n_chunks = len(self.id_to_chunk)

        print "SIZEs (model.py:133) words - chars - tags: ", n_words, n_chars, n_tags
        for f in self.feature_maps:
            print f["name"], len(f["id_to_ftag"])

        # Number of capitalization features
        if cap_dim:
            n_cap = 4

        # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')


        char_for_ids = T.imatrix(name='char_for_ids')
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')

        tag_ids = T.ivector(name='tag_ids')

        tag_ids1 = T.ivector(name='tag_ids1')


        if cap_dim:
            cap_ids = T.ivector(name='cap_ids')

        features_ids = []
        for f in self.feature_maps:
            features_ids.append(T.ivector(name = f['name'] + '_ids'))

        # if pos_dim:
        #     pos_ids = T.ivector(name='pos_ids')
        #
        # if chunk_dim:
        #     chunk_ids = T.ivector(name='chunk_ids')

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            inputs.append(word_input)
            # Initialize with pretrained embeddings
            if pre_emb and training:
                new_weights = word_layer.embeddings.get_value()
                print 'Loading pretrained embeddings from %s...' % pre_emb
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print 'WARNING: %i invalid lines' % emb_invalid
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in xrange(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word.lower()) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word.lower())
                        ]
                        c_zeros += 1
                word_layer.embeddings.set_value(new_weights)
                print 'Loaded %i pretrained embeddings.' % len(pretrained)
                print ('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained embeddings.') % (
                            c_found + c_lower + c_zeros, n_words,
                            100. * (c_found + c_lower + c_zeros) / n_words
                      )
                print ('%i found directly, %i after lowercasing, '
                       '%i after lowercasing + zero.') % (
                          c_found, c_lower, c_zeros
                      )

        #
        # Chars inputs
        #
        if char_dim:
            input_dim += char_lstm_dim
            print "input_dim (model.py:215): ", input_dim
            print "char_lstm_dim (model.py:216): ", char_lstm_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

            char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_rev')

            char_lstm_for.link(char_layer.link(char_for_ids))
            char_lstm_rev.link(char_layer.link(char_rev_ids))

            char_for_output = char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]
            char_rev_output = char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]

            inputs.append(char_for_output)
            if char_bidirect:
                inputs.append(char_rev_output)
                input_dim += char_lstm_dim

        #
        # Capitalization feature
        #
        if cap_dim:
            input_dim += cap_dim
            cap_layer = EmbeddingLayer(n_cap, cap_dim, name='cap_layer')
            inputs.append(cap_layer.link(cap_ids))

        f_layers = []
        for ilayer in range(len(self.feature_maps)):
            f = self.feature_maps[ilayer]
            input_dim += f['dim']
            af_layer = EmbeddingLayer(len(f['id_to_ftag']) , f['dim'], name= f['name'] + '_layer')
            f_layers.append(af_layer)
            inputs.append(af_layer.link(features_ids[ilayer]))

        # Prepare final input
        # if len(inputs) != 1:
        if len(self.tag_maps) > 1:
            inputs1 = inputs * 1
            # inputs1 = T.concatenate(inputs1, axis=1)

        inputs = T.concatenate(inputs, axis=1)

        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)


        ###
        ### layer 0
        ###
        print "input_dim (model.py:332): ", input_dim
        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev')

        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]
        if word_bidirect:
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=1
            )
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_for_output

        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')

            small = -1000
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
            observations = T.concatenate(
                [tags_scores, small * T.ones((s_len, 2))],
                axis=1
            )
            observations = T.concatenate(
                [b_s, observations, e_s],
                axis=0
            )

            # Score from tags
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

            # Score from transitions
            b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
            e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
            padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
            real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()

            all_paths_scores = forward(observations, transitions)
            cost = - (real_path_score - all_paths_scores)

        print "cost: ", cost
        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)


        for af_layer in f_layers:
            self.add_component(af_layer)
            params.extend(af_layer.params)

        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_lstm_for)
            params.extend(char_layer.params)
            params.extend(char_lstm_for.params)
            if char_bidirect:
                self.add_component(char_lstm_rev)
                params.extend(char_lstm_rev.params)

        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)

        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)
        if cap_dim:
            self.add_component(cap_layer)
            params.extend(cap_layer.params)
        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(transitions)
            params.append(transitions)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)

        ###
        ### layer 1
        ###
        if len(self.tag_maps) > 1:
            inputs1.append(tags_scores)
            inputs1 = T.concatenate(inputs1, axis=1)
            print "tags_scores.shape[0]: ", tags_scores.shape[0]
            # input_dim1 = input_dim + int(tags_scores.shape[0])
            # input_dim1 = input_dim + int(tags_scores.shape[0])
            input_dim1 = input_dim + n_tags

            print "input_dim1 (model.py:445): ", input_dim1
            print "input_dim (model.py:332): ", input_dim

            word_lstm_for1 = LSTM(input_dim1, word_lstm_dim, with_batch=False,
                                 name='word_lstm_for1')
            word_lstm_rev1 = LSTM(input_dim1, word_lstm_dim, with_batch=False,
                                 name='word_lstm_rev1')
            word_lstm_for1.link(inputs1)
            word_lstm_rev1.link(inputs1[::-1, :])
            word_for_output1 = word_lstm_for1.h
            word_rev_output1 = word_lstm_rev1.h[::-1, :]
            if word_bidirect:
                final_output1 = T.concatenate(
                    [word_for_output1, word_rev_output1],
                    axis=1
                )
                tanh_layer1 = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                         name='tanh_layer1', activation='tanh')
                final_output1 = tanh_layer1.link(final_output1)
            else:
                final_output1 = word_for_output1

            # Sentence to Named Entity tags - Score
            final_layer1 = HiddenLayer(word_lstm_dim, n_tags1, name='final_layer1',
                                      activation=(None if crf else 'softmax'))
            tags_scores1 = final_layer1.link(final_output1)

            # No CRF
            if not crf:
                cost1 = T.nnet.categorical_crossentropy(tags_scores1, tag_ids1).mean()
            # CRF
            else:
                transitions1 = shared((n_tags1 + 2, n_tags1 + 2), 'transitions1')
                small1 = -1000
                b_s1 = np.array([[small1] * n_tags1 + [0, small1]]).astype(np.float32)
                e_s1 = np.array([[small1] * n_tags1 + [small1, 0]]).astype(np.float32)
                observations1 = T.concatenate(
                    [tags_scores1, small1 * T.ones((s_len, 2))],
                    axis=1
                )
                observations1 = T.concatenate(
                    [b_s1, observations1, e_s1],
                    axis=0
                )

                # Score from tags
                real_path_score1 = tags_scores1[T.arange(s_len), tag_ids1].sum()

                # Score from transitions
                b_id1 = theano.shared(value=np.array([n_tags1], dtype=np.int32))
                e_id1 = theano.shared(value=np.array([n_tags1 + 1], dtype=np.int32))
                padded_tags_ids1 = T.concatenate([b_id1, tag_ids1, e_id1], axis=0)
                real_path_score1 += transitions1[
                    padded_tags_ids1[T.arange(s_len + 1)],
                    padded_tags_ids1[T.arange(s_len + 1) + 1]
                ].sum()

                all_paths_scores1 = forward(observations1, transitions1)
                cost1 = - (real_path_score1 - all_paths_scores1)

                print "cost1 (line 507): ", cost1

            costall = 0.4 * cost + 0.6 * cost1
            print "cost all: ", costall
        else:
            costall = cost

        if len(self.tag_maps) > 1:
            self.add_component(word_lstm_for1)
            params.extend(word_lstm_for1.params)

            if word_bidirect:
                self.add_component(word_lstm_rev1)
                params.extend(word_lstm_rev1.params)

            self.add_component(final_layer1)
            params.extend(final_layer1.params)

            if crf:
                self.add_component(transitions1)
                params.append(transitions1)

            if word_bidirect:
                self.add_component(tanh_layer1)
                params.extend(tanh_layer1.params)

        # Prepare train and eval inputs
        eval_inputs = []

        if word_dim:
            eval_inputs.append(word_ids)

        for ilayer in range(len(self.feature_maps)):
            eval_inputs.append(features_ids[ilayer])


        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)

        if cap_dim:
            eval_inputs.append(cap_ids)

        if len(self.tag_maps) > 1:
            train_inputs = eval_inputs + [tag_ids] + [tag_ids1]
        else:
            train_inputs = eval_inputs + [tag_ids]

        print "train_inputs: ", train_inputs

        test_value = train_inputs[0].shape
        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print 'Compiling...'
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, costall, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=costall,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )

            # f_test = theano.function(
            #     inputs=train_inputs,
            #     outputs=[tags_scores.shape, word_ids.shape, tag_ids.shape, char_for_ids.shape, word_for_output.shape],
            #     givens=({is_train: np.cast['int32'](1)} if dropout else {}),
            #     on_unused_input='warn'
            # )
        else:
            f_train = None

        # Compile evaluation function
        if len(self.tag_maps) > 1:
            tags_scores_out = [tags_scores, tags_scores1]
        else:
            tags_scores_out = [tags_scores]

        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores_out,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            if (len(self.tag_maps) == 1):
                f_eval = theano.function(
                    inputs=eval_inputs,
                    outputs=forward(observations, transitions, viterbi=True,
                                    return_alpha=False, return_best_sequence=True),
                    givens=({is_train: np.cast['int32'](0)} if dropout else {})
                )
            else:
                f_eval = theano.function(
                    inputs=eval_inputs,
                    outputs=forward1(observations, transitions, observations1, transitions1, viterbi=True,
                                    return_alpha=False, return_best_sequence=True),
                    givens=({is_train: np.cast['int32'](0)} if dropout else {})
                )

        from pprint import pprint
        pprint (self.components)
        # return f_train, f_eval, f_test
        return f_train, f_eval
