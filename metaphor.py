#!/usr/bin/env python2.7
from __future__ import division
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import theano
import theano.tensor as T
import lasagne
import codecs
import time
from sklearn.metrics import f1_score
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

m = 1
l = 0


# prepare the batch for the single sequence
def prepare_single_batch(batch_size, x, mask, y, shuffle=True):
    x_batches = []
    mask_batches = []
    y_batches = []
    n = len(x)
    sidx = np.arange(n, dtype='int32')
    if shuffle:
        np.random.shuffle(sidx)
    batches = []
    start = 0
    for i in range(n // batch_size):
        batches.append(sidx[start:start + batch_size])
        start += batch_size
    if start != n:
        batches.append(sidx[start:])
    for batch in batches:
        x_batch = []
        mask_batch = []
        y_batch = []
        for s in batch:
            x_batch.append(x[s])
            if len(mask):
                mask_batch.append(mask[s])
            y_batch.append(y[s])
        x_batches.append(x_batch)
        mask_batches.append(mask_batch)
        y_batches.append(y_batch)
    return x_batches, mask_batches, y_batches


# prepare the batch for multiple sequences
def prepare_multi_batch(batch_size, x_dep, x_svo, mask_svo, x_text, y, shuffle=True):
    x_dep_batches = []
    x_svo_batches = []
    x_text_batches = []
    x_mask_batches = []
    y_batches = []
    n = len(x_svo)
    sidx = np.arange(n, dtype='int32')
    if shuffle:
        np.random.shuffle(sidx)
    batches = []
    start = 0
    for i in range(n // batch_size):
        batches.append(sidx[start:start + batch_size])
        start += batch_size
    if start != n:
        batches.append(sidx[start:])
    for batch in batches:
        x_dep_batch = []
        x_text_batch = []
        x_svo_batch = []
        x_mask_batch = []
        y_batch = []
        for s in batch:
            x_dep_batch.append(x_dep[s])
            x_text_batch.append(x_text[s])
            x_svo_batch.append(x_svo[s])
            x_mask_batch.append(mask_svo[s])
            y_batch.append(y[s])
        x_dep_batches.append(x_dep_batch)
        x_text_batches.append(x_text_batch)
        x_svo_batches.append(x_svo_batch)
        x_mask_batches.append(x_mask_batch)
        y_batches.append(y_batch)
    return x_dep_batches, x_svo_batches, x_mask_batches, x_text_batches, y_batches


# padding the batch
def padding(batch, w_dim):
    max_len = 0
    sens_lens = []
    for sen in batch:
        l_sen = len(sen)
        sens_lens.append(l_sen)
        if max_len < l_sen:
            max_len = l_sen
    if max_len == 0:
        return None, None
    x_padded = np.zeros((len(batch), max_len, w_dim), dtype='float32')
    x_mask = np.zeros((len(batch), max_len), dtype='int32')
    for idx, sen in enumerate(batch):
        sen_len = sens_lens[idx]
        x_padded[idx, :sen_len, :] = sen
        x_mask[idx, :sen_len] = 1
    return x_padded, x_mask


# data structure for SVO sequence
class Svo:
    def __init__(self, line, label):
        self.label = label
        line = line.strip().split("\t")
        if len(line) == 2:
            instance = line[0]
            svo = instance.split("_")
            self.id = int(svo[0])
            self.s_id = int(svo[3])
            self.s_lemma = svo[4]
            self.s_surface_form = svo[5]
            self.v_id = int(svo[6])
            self.v_lemma = svo[7]
            self.v_surface_form = svo[8]
            self.o_id = int(svo[9])
            self.o_lemma = svo[10]
            self.o_surface_form = svo[11]
        else:
            self.s = line[0]
            self.v = line[1]
            self.o = line[2]


# data structure for Dep word
class DepWord:
    def __init__(self, line):
        tokens = line.strip().replace(u'\uFFFD', '?').split('\t')
        # u'\uFFFD' represent an unknown or unrepresentable character
        self.id = int(tokens[0]) - 1
        self.surface_form = tokens[1].split()[-1].lower()  # Only the last word.
        self.pos_tag = tokens[3]
        self.head = int(tokens[6]) - 1
        self.dep_rel = tokens[7]
        self.lemma = tokens[2].lower()
        if self.lemma in [u'-', u'_']:
            if self.pos_tag.startswith("V"):
                lemmatizer_pos = 'v'
            elif self.pos_tag.startswith("N"):
                lemmatizer_pos = 'n'
            else:
                lemmatizer_pos = 'a'
            self.lemma = lemmatizer.lemmatize(self.surface_form, lemmatizer_pos).lower()


# data structure for Dep sequence
class DepSentence:
    def __init__(self):
        self.words = []

    def add_word(self, line):
        word = DepWord(line)
        self.words.append(word)

    def get_dep_seq(self, verb):
        sequence = [verb]
        for word in self.words:
            if word.head == verb.id:
                sequence.append(word)
            elif word.id == verb.head:
                sequence.append(word)

        def word_cmp(x, y):
            return x.id - y.id

        sorted(sequence, cmp=word_cmp)

        return sequence

    def clear(self):
        self.words = []


# LSTM Model
class LstmModel:
    def __init__(self, word2vecFile):
        self.word2vec_file = word2vecFile
        self.batch_size = 30
        self.num_epoch = 100
        self.learning_rate = 0.002
        self.dropout_rate = 0.6
        self.train_portion = 0.9
        self.num_target = 2

        self.word2vecModel = KeyedVectors.load_word2vec_format(self.word2vec_file, binary=True)
        self.num_words = len(self.word2vecModel.vocab)
        self.none = self.word2vecModel.index2word[0]
        self.w_dim = self.word2vecModel[self.none].shape[0]

    def load_svo_train_data(self, metaphorical_svo_file, literal_svo_file):
        def load(svo_file, svo_label):
            x = []
            y = []
            mask = []
            svo_all = []
            for line in codecs.open(svo_file, "r", "utf-8"):
                svo = Svo(line, svo_label)
                x_temp = []
                mask_temp = []
                if svo.s_id == 999 or (
                                svo.s_lemma not in self.word2vecModel and svo.s_surface_form not in self.word2vecModel):
                    mask_temp.append(0)
                    x_temp.append(self.word2vecModel[self.none])
                else:
                    mask_temp.append(1)
                    x_temp.append(
                        self.word2vecModel[svo.s_surface_form] if svo.s_surface_form in self.word2vecModel else
                        self.word2vecModel[svo.s_lemma])
                if svo.v_id == 999 or (
                                svo.v_lemma not in self.word2vecModel and svo.v_surface_form not in self.word2vecModel):
                    mask_temp.append(0)
                    x_temp.append(self.word2vecModel[self.none])
                else:
                    mask_temp.append(1)
                    x_temp.append(
                        self.word2vecModel[svo.v_surface_form] if svo.v_surface_form in self.word2vecModel else
                        self.word2vecModel[svo.v_lemma])
                if svo.o_id == 999 or (
                                svo.o_lemma not in self.word2vecModel and svo.o_surface_form not in self.word2vecModel):
                    mask_temp.append(0)
                    x_temp.append(self.word2vecModel[self.none])
                else:
                    mask_temp.append(1)
                    x_temp.append(
                        self.word2vecModel[svo.o_surface_form] if svo.o_surface_form in self.word2vecModel else
                        self.word2vecModel[svo.o_lemma])
                assert len(mask_temp) == 3
                if mask_temp.count(1) > 1 and mask_temp[1] == 1:
                    x.append(x_temp)
                    mask.append(mask_temp)
                    y.append(svo_label)
                    svo_all.append(svo)
            return x, y, mask, svo_all

        m_x, m_y, m_mask, m_svo = load(metaphorical_svo_file, m)
        l_x, l_y, l_mask, l_svo = load(literal_svo_file, l)

        print "    metaphorical svo size:   ", len(m_y)
        print "    literal svo size:        ", len(l_y)

        return (m_x + l_x), (m_y + l_y), (m_mask + l_mask), m_svo, l_svo

    def load_train_text_dep_data(self, metaphorical_dep_file, literal_dep_file, m_svo, l_svo):
        def load(dep_file, svo):
            text_x = []
            dep_x = []
            seq_num = 0
            idx = 0
            sentence = DepSentence()
            for line in codecs.open(dep_file, "r", "utf-8"):
                if idx > len(svo):
                    break
                if len(line.strip()) == 0:
                    while idx < len(svo) and svo[idx].id == seq_num:
                        x_text_temp = []
                        x_dep_temp = []
                        sequence = sentence.get_dep_seq(sentence.words[svo[idx].v_id])
                        for w in sequence:
                            if w.surface_form in self.word2vecModel or w.lemma in self.word2vecModel:
                                x_dep_temp.append(
                                    self.word2vecModel[w.surface_form]
                                    if w.surface_form in self.word2vecModel else self.word2vecModel[w.lemma])
                        for w in sentence.words:
                            if w.surface_form in self.word2vecModel or w.lemma in self.word2vecModel:
                                x_text_temp.append(
                                    self.word2vecModel[w.surface_form]
                                    if w.surface_form in self.word2vecModel else self.word2vecModel[w.lemma])
                        text_x.append(x_text_temp)
                        dep_x.append(x_dep_temp)
                        idx += 1
                    seq_num += 1
                    sentence.clear()
                elif svo[idx].id == seq_num:
                    sentence.add_word(line)
            return text_x, dep_x

        m_text, m_dep = load(metaphorical_dep_file, m_svo)
        l_text, l_dep = load(literal_dep_file, l_svo)

        print "    metaphorical text size:  ", len(m_text)
        print "    literal text size:       ", len(l_text)
        print "    metaphorical dep size:   ", len(m_dep)
        print "    literal dep size:        ", len(l_dep)

        return (m_text + l_text), (m_dep + l_dep)

    def load_train_data(self):
        print "Load train data ..."
        metaphorical_dep_file = 'data/train/metaphorical.dep'
        literal_dep_file = 'data/train/literal.dep'
        metaphorical_svo_file = 'data/train/metaphorical.svo'
        literal_svo_file = 'data/train/literal.svo'

        svo_x, y, svo_mask, m_svo, l_svo = self.load_svo_train_data(metaphorical_svo_file, literal_svo_file)
        text_x, dep_x = self.load_train_text_dep_data(metaphorical_dep_file, literal_dep_file, m_svo, l_svo)

        print "-----------------------------------------------------"

        train_n = int(np.round(len(y) * self.train_portion))
        sidx = np.random.permutation(len(y))

        self.dep_train = [dep_x[s] for s in sidx[:train_n]]
        self.svo_train = [svo_x[s] for s in sidx[:train_n]]
        self.text_train = [text_x[s] for s in sidx[:train_n]]
        self.y_train = [y[s] for s in sidx[:train_n]]

        self.dep_valid = [dep_x[s] for s in sidx[train_n:]]
        self.svo_valid = [svo_x[s] for s in sidx[train_n:]]
        self.text_valid = [text_x[s] for s in sidx[train_n:]]
        self.y_valid = [y[s] for s in sidx[train_n:]]

        self.mask_svo_train = [svo_mask[s] for s in sidx[:train_n]]
        self.mask_svo_valid = [svo_mask[s] for s in sidx[train_n:]]

        print "    train set size:   ", len(self.y_train)
        print "    valid set size:   ", len(self.y_valid)
        print "    train set metaphorical size:  ", self.y_train.count(m)
        print "    train set literal size:       ", self.y_train.count(l)
        print "    valid set metaphorical size:  ", self.y_valid.count(m)
        print "    valid set literal size:       ", self.y_valid.count(l)
        print "-----------------------------------------------------"

    def load_test_data(self):
        print "Load test data ..."
        metaphorical_dep_file = 'data/test/metaphorical.dep'
        literal_dep_file = 'data/test/literal.dep'
        metaphorical_svo_file = 'data/test/metaphorical.svo'
        literal_svo_file = 'data/test/literal.svo'

        m_lemma2word = ["0_see_saw", "16_Texan_Texans", "20_Hawaii_Hawaii",
                        "31_Tori_Tori", "37_Murkowski_Murkowski", "39_man_men",
                        "45_stare_staring", "55_Bundestag_Bundestag", "63_midwesterner_Midwesterners",
                        "78_Bernake_Bernake", "79_Apple_Apple", "97_Mitt Romney_Mitt Romney", "99_Olmert_Olmert",
                        ]
        l_lemma2word = ["16_stairs_stairs", "18_Jim_Jim", "36_feed_feed", "42_vegetables_vegetables",
                        "48_person_people", "51_Iran_Iran", "57_person_People", "62_Miranda_Miranda",
                        "63_antena_antennas", "75_Hawaii_Hawaii", "83_Dean_Dean", "84_Frodo_Frodo",
                        "87_Frodo_Frodo", "87_stare_stared", "90_Blair_Blair", "91_Beth_Beth",
                        "92_Frank_Frank", "101_Lebanese_Lebanese", "105_Jim_jim", "107_Misha_Misha"
                        ]

        def load_svo(svo_file):
            svo = []
            for line in codecs.open(svo_file, "r", "utf-8"):
                words = line.strip().split("\t")
                svo.append(words)
            return svo

        m_svo = load_svo(metaphorical_svo_file)
        l_svo = load_svo(literal_svo_file)

        def load_test_text_dep_data(dep_file, svo, l2w, label):
            y = []
            text_x = []
            dep_x = []
            svo_x = []
            mask_x = []
            sentence = DepSentence()
            sen_num = 0
            l2w_num = 0
            for line in codecs.open(dep_file, "r", 'utf-8'):
                if len(line.strip()) == 0:
                    temp_text = []
                    for w in sentence.words:
                        if w.surface_form in self.word2vecModel or w.lemma in self.word2vecModel:
                            temp_text.append(
                                self.word2vecModel[w.surface_form]
                                if w.surface_form in self.word2vecModel else self.word2vecModel[w.lemma])

                    temp_mask = []
                    temp_svo = []
                    temp_dep = []
                    w_num = 0
                    for w in svo[sen_num]:
                        if w == "none":
                            temp_mask.append(0)
                            temp_svo.append(self.word2vecModel[self.none])
                            w_num += 1
                            continue

                        if w_num == 1:
                            verb = None
                            if l2w_num < len(l2w):
                                l2wst = l2w[l2w_num].split("_")
                                temp_num = int(l2wst[0])

                                if sen_num == temp_num and w == l2wst[1]:
                                    for word in sentence.words:
                                        if word.surface_form == l2wst[2]:
                                            verb = word
                            if not verb:
                                for word in sentence.words:
                                    if word.lemma == w:
                                        verb = word
                            seq = sentence.get_dep_seq(verb)
                            for se in seq:
                                if se.surface_form in self.word2vecModel or se.lemma in self.word2vecModel:
                                    temp_dep.append(
                                        self.word2vecModel[se.surface_form]
                                        if se.surface_form in self.word2vecModel else self.word2vecModel[se.lemma])

                        if l2w_num < len(l2w):
                            st = l2w[l2w_num].split("_")
                            num = int(st[0])
                            if sen_num == num and w == st[1]:
                                if st[2] in self.word2vecModel or st[1] in self.word2vecModel:
                                    temp_svo.append(
                                        self.word2vecModel[st[2]]
                                        if st[2] in self.word2vecModel else self.word2vecModel[st[1]])
                                    temp_mask.append(1)
                                else:
                                    temp_mask.append(0)
                                    temp_svo.append(self.word2vecModel[self.none])
                                l2w_num += 1
                                w_num += 1
                                continue

                        for word in sentence.words:
                            if word.lemma == w:
                                if word.surface_form in self.word2vecModel or word.lemma in self.word2vecModel:
                                    temp_svo.append(
                                        self.word2vecModel[word.surface_form]
                                        if word.surface_form in self.word2vecModel else self.word2vecModel[word.lemma])
                                    temp_mask.append(1)
                                else:
                                    temp_mask.append(0)
                                    temp_svo.append(self.word2vecModel[self.none])

                        w_num += 1
                    if temp_mask.count(1) > 1 and temp_mask[1] == 1:
                        text_x.append(temp_text)
                        dep_x.append(temp_dep)
                        svo_x.append(temp_svo)
                        mask_x.append(temp_mask)
                        y.append(label)
                    sen_num += 1
                    sentence.clear()
                else:
                    sentence.add_word(line)
            return text_x, dep_x, svo_x, mask_x, y

        m_text, m_dep, m_svo_test, m_mask, m_y = load_test_text_dep_data(metaphorical_dep_file, m_svo, m_lemma2word, m)
        l_text, l_dep, l_svo_test, l_mask, l_y = load_test_text_dep_data(literal_dep_file, l_svo, l_lemma2word, l)

        self.y_test = m_y + l_y
        self.text_test = m_text + l_text
        self.svo_test = m_svo_test + l_svo_test
        self.dep_test = m_dep + l_dep
        self.mask_svo_test = m_mask + l_mask

        print "    test set size:    ", len(self.y_test)
        print "    test set metaphorical size:  ", self.y_test.count(m)
        print "    test set literal size:       ", self.y_test.count(l)
        print "-----------------------------------------------------"

    def build_single_model(self, in_var, mask_var, unit_num):
        input_in = lasagne.layers.InputLayer(shape=(None, None, self.w_dim), input_var=in_var)
        input_in = lasagne.layers.DropoutLayer(input_in, self.dropout_rate)
        mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)
        llstm = lasagne.layers.LSTMLayer(incoming=input_in, num_units=unit_num, mask_input=mask,
                                         name='llstm', only_return_final=True)
        rlstm = lasagne.layers.LSTMLayer(incoming=input_in, num_units=unit_num, mask_input=mask,
                                         name='rlstm', only_return_final=True, backwards=False)
        llstm = lasagne.layers.DropoutLayer(llstm, self.dropout_rate)
        rlstm = lasagne.layers.DropoutLayer(rlstm, self.dropout_rate)
        lstm = lasagne.layers.ConcatLayer([llstm, rlstm], axis=-1)
        out_model = lasagne.layers.DenseLayer(lstm, num_units=self.num_target,
                                              nonlinearity=lasagne.nonlinearities.softmax)
        return out_model

    def build_multi_model(self, sentence_var, mask_sen_var, dep_var, mask_dep_var, svo_var, mask_svo_var):
        sentence_in = lasagne.layers.InputLayer(shape=(None, None, self.w_dim), input_var=sentence_var)
        sentence_in = lasagne.layers.DropoutLayer(sentence_in, self.dropout_rate)
        sen_l_mask_var = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_sen_var)
        sen_llstm = lasagne.layers.LSTMLayer(incoming=sentence_in, num_units=150, mask_input=sen_l_mask_var,
                                             name='sen_llstm', only_return_final=True)
        sen_rlstm = lasagne.layers.LSTMLayer(incoming=sentence_in, num_units=150, mask_input=sen_l_mask_var,
                                             name='sen_rlstm', only_return_final=True, backwards=False)
        sen_llstm = lasagne.layers.DropoutLayer(sen_llstm, self.dropout_rate)
        sen_rlstm = lasagne.layers.DropoutLayer(sen_rlstm, self.dropout_rate)

        dep_in = lasagne.layers.InputLayer(shape=(None, None, self.w_dim), input_var=dep_var)
        dep_in = lasagne.layers.DropoutLayer(dep_in, self.dropout_rate)
        dep_l_mask_var = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_dep_var)
        dep_llstm = lasagne.layers.LSTMLayer(incoming=dep_in, num_units=60, mask_input=dep_l_mask_var,
                                             name='dep_llstm', only_return_final=True)
        dep_rlstm = lasagne.layers.LSTMLayer(incoming=dep_in, num_units=60, mask_input=dep_l_mask_var,
                                             name='dep_rlstm', only_return_final=True, backwards=False)
        dep_llstm = lasagne.layers.DropoutLayer(dep_llstm, self.dropout_rate)
        dep_rlstm = lasagne.layers.DropoutLayer(dep_rlstm, self.dropout_rate)

        svo_in = lasagne.layers.InputLayer(shape=(None, None, self.w_dim), input_var=svo_var)
        svo_in = lasagne.layers.DropoutLayer(svo_in, self.dropout_rate)
        svo_l_mask_var = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_svo_var)
        svo_llstm = lasagne.layers.LSTMLayer(incoming=svo_in, num_units=40, mask_input=svo_l_mask_var,
                                             name='svo_llstm', only_return_final=True)
        svo_rlstm = lasagne.layers.LSTMLayer(incoming=svo_in, num_units=40, mask_input=svo_l_mask_var,
                                             name='svo_rlstm', only_return_final=True, backwards=False)
        svo_llstm = lasagne.layers.DropoutLayer(svo_llstm, self.dropout_rate)
        svo_rlstm = lasagne.layers.DropoutLayer(svo_rlstm, self.dropout_rate)

        con_in = lasagne.layers.ConcatLayer([sen_llstm, sen_rlstm, dep_llstm, dep_rlstm, svo_llstm, svo_rlstm], axis=-1)

        con_lstm = lasagne.layers.DenseLayer(con_in, num_units=200)
        output = lasagne.layers.DenseLayer(con_lstm, num_units=self.num_target,
                                           nonlinearity=lasagne.nonlinearities.softmax)
        return output

    def build_single_model_fn(self, unit_num):
        in_var = T.ftensor3('in_var')
        mask_var = T.imatrix('mask_var')
        network = self.build_single_model(in_var, mask_var, unit_num)
        prediction = lasagne.layers.get_output(network)
        target_var = T.ivector('targets')
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        all_params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.momentum(loss, all_params, learning_rate=self.learning_rate)

        valid_prediction = lasagne.layers.get_output(network, deterministic=True)
        valid_loss = lasagne.objectives.categorical_crossentropy(valid_prediction, target_var)
        valid_loss = valid_loss.mean()

        valid_acc = lasagne.objectives.categorical_accuracy(valid_prediction, target_var)
        predicted_label = T.argmax(valid_prediction, axis=1)
        valid_acc = T.mean(valid_acc)

        self.single_train_fn = theano.function([in_var, mask_var, target_var], loss, updates=updates)
        self.single_val_fn = theano.function([in_var, mask_var, target_var], [valid_loss, valid_acc])
        self.single_prediction_fn = theano.function([in_var, mask_var], predicted_label)

    def build_multi_model_fn(self):
        sentence_var = T.ftensor3('sentence_var')
        svo_var = T.ftensor3('svo_var')
        dep_var = T.ftensor3('dep_var')
        mask_sen_var = T.imatrix('mask_sen_var')
        mask_dep_var = T.imatrix('mask_dep_var')
        mask_svo_var = T.imatrix('mask_svo_var')
        network = self.build_multi_model(sentence_var, mask_sen_var, dep_var, mask_dep_var, svo_var, mask_svo_var)
        prediction = lasagne.layers.get_output(network)
        target_var = T.ivector('targets')
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        all_params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.momentum(loss, all_params, learning_rate=self.learning_rate)

        valid_prediction = lasagne.layers.get_output(network, deterministic=True)
        valid_loss = lasagne.objectives.categorical_crossentropy(valid_prediction, target_var)
        valid_loss = valid_loss.mean()

        valid_acc = lasagne.objectives.categorical_accuracy(valid_prediction, target_var)
        predicted_label = T.argmax(valid_prediction, axis=1)
        valid_acc = T.mean(valid_acc)

        self.multi_train_fn = theano.function([
            sentence_var, mask_sen_var, dep_var, mask_dep_var, svo_var, mask_svo_var, target_var],
            loss, updates=updates)
        self.multi_val_fn = theano.function([
            sentence_var, mask_sen_var, dep_var, mask_dep_var, svo_var, mask_svo_var, target_var],
            [valid_loss, valid_acc])
        self.multi_prediction_fn = theano.function([
            sentence_var, mask_sen_var, dep_var, mask_dep_var, svo_var, mask_svo_var],
            predicted_label)

    def train_single_model(self, model_type, units_num):
        self.build_single_model_fn(units_num)
        mask_train = []
        mask_valid = []
        mask_test = []

        if model_type == "svo":
            x_train = self.svo_train
            x_valid = self.svo_valid
            x_test = self.svo_test
            mask_train = self.mask_svo_train
            mask_valid = self.mask_svo_valid
            mask_test = self.mask_svo_test
        elif model_type == "text":
            x_train = self.text_train
            x_valid = self.text_valid
            x_test = self.text_test
        else:
            x_train = self.dep_train
            x_valid = self.dep_valid
            x_test = self.dep_test

        print "Training ..."
        for epoch in range(self.num_epoch):
            x_train_batches, mask_train_batches, y_train_batches = prepare_single_batch(
                self.batch_size, x_train, mask_train, self.y_train, shuffle=True)
            train_err = 0
            train_batches_num = 0
            train_acc = 0
            start_time = time.time()
            for i in range(len(y_train_batches)):
                if model_type != "svo":
                    x_train_batch, mask_train_batch = padding(x_train_batches[i], self.w_dim)
                else:
                    x_train_batch = x_train_batches[i]
                    mask_train_batch = mask_train_batches[i]
                x_train_array = np.asarray(x_train_batch, dtype="float32")
                mask_train_array = np.asarray(mask_train_batch, dtype="int32")
                y_train_array = np.asarray(y_train_batches[i], dtype='int32')
                train_err += self.single_train_fn(x_train_array, mask_train_array, y_train_array)
                err, acc = self.single_val_fn(x_train_array, mask_train_array, y_train_array)
                train_acc += acc
                train_batches_num += 1
            print "  Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.num_epoch, time.time() - start_time)
            print "    training loss:        {:.6f}".format(train_err / train_batches_num)
            print "    training accuracy:    {:.2f} %".format(train_acc / train_batches_num * 100)

            val_err = 0
            val_acc = 0
            val_batches_num = 0

            x_valid_batches, mask_valid_batches, y_valid_batches = prepare_single_batch(
                self.batch_size, x_valid, mask_valid, self.y_valid, shuffle=False)
            for i in range(len(y_valid_batches)):
                if model_type != "svo":
                    x_valid_batch, mask_valid_batch = padding(x_valid_batches[i], self.w_dim)
                else:
                    x_valid_batch = x_valid_batches[i]
                    mask_valid_batch = mask_valid_batches[i]
                x_valid_array = np.asarray(x_valid_batch, dtype="float32")
                mask_valid_array = np.asarray(mask_valid_batch, dtype="int32")
                y_valid_array = np.asarray(y_valid_batches[i], dtype='int32')

                err, acc = self.single_val_fn(
                    x_valid_array, mask_valid_array, y_valid_array)
                val_err += err
                val_acc += acc
                val_batches_num += 1
            print "    validation loss:        {:.6f}".format(val_err / val_batches_num)
            print "    validation accuracy:    {:.2f} %".format(val_acc / val_batches_num * 100)

            labels = []
            test_err = 0
            test_acc = 0
            test_batches_num = 0
            x_test_batches, mask_test_batches, y_test_batches = prepare_single_batch(
                self.batch_size, x_test, mask_test, self.y_test, shuffle=False)
            for i in range(len(y_test_batches)):
                if model_type != "svo":
                    x_test_batch, mask_test_batch = padding(x_test_batches[i], self.w_dim)
                else:
                    x_test_batch = x_test_batches[i]
                    mask_test_batch = mask_test_batches[i]
                x_test_array = np.asarray(x_test_batch, dtype="float32")
                mask_test_array = np.asarray(mask_test_batch, dtype="int32")
                y_test_array = np.asarray(y_test_batches[i], dtype='int32')

                err, acc = self.single_val_fn(
                    x_test_array, mask_test_array, y_test_array
                )
                batch_labels = self.single_prediction_fn(
                    x_test_array, mask_test_array)
                labels += list(batch_labels)
                test_err += err
                test_acc += acc
                test_batches_num += 1
            assert len(labels) == len(self.y_test)
            print "    Test loss:        {:.6f}".format(test_err / test_batches_num)
            print "    Test accuracy:    {:.2f} %".format(test_acc / test_batches_num * 100)
            print "    F-score:          {:.2f}".format(f1_score(list(self.y_test), labels))
            print "-----------------------------------------------------"

    def train_multi_model(self):
        self.build_multi_model_fn()
        print "Training ..."
        for epoch in range(self.num_epoch):
            x_dep_batches, x_svo_batches, x_mask_svo_batches, x_text_batches, y_batches = \
                prepare_multi_batch(
                    self.batch_size, self.dep_train,
                    self.svo_train, self.mask_svo_train, self.text_train, self.y_train, shuffle=True)
            train_err = 0
            train_batches_num = 0
            train_acc = 0
            start_time = time.time()
            for x_dep_batch, x_svo_batch, x_mask_batch, x_text_batch, y_batch in zip(
                    x_dep_batches, x_svo_batches, x_mask_svo_batches, x_text_batches, y_batches):
                dep_train, dep_mask = padding(x_dep_batch, self.w_dim)
                text_train, text_mask = padding(x_text_batch, self.w_dim)
                svo_train = np.asarray(x_svo_batch, dtype="float32")
                svo_mask = np.asarray(x_mask_batch, dtype="int32")
                y_train = np.asarray(y_batch, dtype='int32')
                train_err += self.multi_train_fn(
                    text_train, text_mask, dep_train, dep_mask, svo_train, svo_mask, y_train)
                err, acc = self.multi_val_fn(text_train, text_mask, dep_train, dep_mask, svo_train, svo_mask, y_train)
                train_acc += acc
                train_batches_num += 1
            print "  Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.num_epoch, time.time() - start_time)
            print "    training loss:        {:.6f}".format(train_err / train_batches_num)
            print "    training accuracy:    {:.2f} %".format(train_acc / train_batches_num * 100)

            val_err = 0
            val_acc = 0
            val_batches_num = 0

            x_dep_batches_valid, x_svo_batches_valid, x_svo_mask_batches_valid, x_text_batches_valid, y_batches_valid \
                = prepare_multi_batch(
                 self.batch_size, self.dep_valid, self.svo_valid,
                 self.mask_svo_valid, self.text_valid, self.y_valid, shuffle=False)
            for x_dep_batch, x_svo_batch, x_svo_mask_batch, x_text_batch, y_valid_batch \
                    in zip(x_dep_batches_valid, x_svo_batches_valid,
                           x_svo_mask_batches_valid, x_text_batches_valid, y_batches_valid):
                x_dep_valid, mask_dep_valid = padding(x_dep_batch, self.w_dim)
                x_text_valid, mask_text_valid = padding(x_text_batch, self.w_dim)
                x_svo_valid = np.asarray(x_svo_batch, dtype="float32")
                mask_svo_valid = np.asarray(x_svo_mask_batch, dtype="int32")
                y_valid = np.asarray(y_valid_batch, dtype='int32')
                err, acc = self.multi_val_fn(
                    x_text_valid, mask_text_valid, x_dep_valid, mask_dep_valid, x_svo_valid, mask_svo_valid, y_valid)
                val_err += err
                val_acc += acc
                val_batches_num += 1
            print "    validation loss:        {:.6f}".format(val_err / val_batches_num)
            print "    validation accuracy:    {:.2f} %".format(val_acc / val_batches_num * 100)

            labels = []
            test_err = 0
            test_acc = 0
            test_batches_num = 0
            test_dep_batches, test_svo_batches, test_svo_mask_batches, test_text_batches, test_y_batches \
                = prepare_multi_batch(
                 self.batch_size, self.dep_test, self.svo_test,
                 self.mask_svo_test, self.text_test, self.y_test, shuffle=False)
            for test_dep_batch, test_svo_batch, test_svo_mask_batch, test_text_batch, test_y_batch in zip(
                    test_dep_batches, test_svo_batches, test_svo_mask_batches, test_text_batches, test_y_batches):
                test_dep_, test_dep_mask = padding(test_dep_batch, self.w_dim)
                test_text_, test_text_mask = padding(test_text_batch, self.w_dim)
                test_svo_ = np.asarray(test_svo_batch, dtype="float32")
                test_svo_mask = np.asarray(test_svo_mask_batch, dtype="int32")
                test_y_ = np.asarray(test_y_batch, dtype='int32')
                err, acc = self.multi_val_fn(
                    test_text_, test_text_mask, test_dep_, test_dep_mask, test_svo_, test_svo_mask, test_y_)
                batch_labels = self.multi_prediction_fn(
                    test_text_, test_text_mask, test_dep_, test_dep_mask, test_svo_, test_svo_mask)
                labels += list(batch_labels)
                test_err += err
                test_acc += acc
                test_batches_num += 1
            assert len(labels) == len(self.y_test)
            print "    Test loss:        {:.6f}".format(test_err / test_batches_num)
            print "    Test accuracy:    {:.2f} %".format(test_acc / test_batches_num * 100)
            print "    F-score:          {:.2f}".format(f1_score(list(self.y_test), labels))
            print "-----------------------------------------------------"


if __name__ == '__main__':
    word2vec_file = 'data/GoogleNews-vectors-negative300.bin'

    model = LstmModel(word2vec_file)
    model.load_train_data()
    model.load_test_data()
    # train and test LSTM model for multiple sequence
    model.train_multi_model()

    """
    # train and test LSTM model for single sequence like svo and unit_num = 128
    model.train_single_model("svo", 128)
    """
