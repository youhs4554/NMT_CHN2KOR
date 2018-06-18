#-*- coding: utf-8 -*-

import pandas as pd
import pickle
from collections import Counter
import re
from konlpy.tag import Kkma
import tqdm
import numpy as np



class Dataset(object):
    def __init__(self, src_file, target_file):
        self.word2ix = pickle.load(file('./data/word2ix.pkl'))
        self.chunking(src_file, target_file)
        # convert word to ix
        for n, (src_sent, target_sent) in enumerate(self.samples):
            src_ixs = []
            target_ixs = []
            for w in src_sent.split():
                w = '<unk>' if w not in self.word2ix['src'] else w
                src_ixs.append(self.word2ix['src'][w])

            for w in target_sent.split():
                w = '<unk>' if w not in self.word2ix['target'] else w
                target_ixs.append(self.word2ix['target'][w])

            self.samples[n] = dict(src_sent=src_sent,src_ixs=src_ixs, target_sent=target_sent, target_ixs=target_ixs)

        self.n = len(self.samples)

        # save datasets
        pickle.dump(self.samples, file('./data_eval/samples.pkl', 'wb'))

    def chunking(self, src_file, target_file):
        # initialize nlp tool
        kkm = Kkma()

        self.fd_src = file(src_file)
        self.fd_target = file(target_file)

        self.samples = []  # list of (src, target) tuple
        self.words_list = dict(src_words=[], target_words=[])

        src_lines = self.fd_src.readlines()
        target_lines = self.fd_target.readlines()

        assert len(src_lines)==len(target_lines)

        for i in tqdm.tqdm(range(len(src_lines))):
            src_sent = src_lines[i].strip().decode('utf-8')
            target_sent = target_lines[i].strip().decode('utf-8')

            # extract named_entities for source language
            named_entities = [tag[0] for tag in kkm.pos(target_sent) if
                              tag[1] == 'OH']  # extract named-entities from target sentence

            # preprocess target_sent(remove comments and any other noise)
            # target_sent = re.sub(r'\d*].*', '', target_sent)
            # target_sent = re.sub(r'\([^)]*\)', '', target_sent)
            # target_sent = re.sub(r'\d*\)', '', target_sent)

            # src/target pos-tags
            src_tags = kkm.pos(src_sent)
            target_tags = kkm.pos(target_sent)

            # symbols of target language
            symbols = [ tag[0] for tag in target_tags if tag[1].startswith('S') ]

            src_sent = ' '.join([tag[0] for tag in src_tags if not tag[1].startswith('S')])  # remove all symbols

            target_sent = ' '.join([tag[0] for tag in target_tags if not tag[1].startswith('S')])

            for sym in symbols:
                target_sent = target_sent.replace(sym, '') # remove symbol

            for ne in named_entities:
                src_sent = src_sent.replace(ne, '*' + ne + '*')
                # target_sent = target_sent.replace('( '+ne, '( '+ne+' )')
                target_sent = target_sent.replace(ne, '') # remove hanja

            src_sent = ' '.join(src_sent.split('*'))

            if not src_sent or not target_sent:
                continue

            if len(src_sent.split()) > 100 or len(target_sent.split()) > 100:
               continue

            cur_src_words = []
            for word in src_sent.split():
                if word:
                    if word in named_entities:
                        self.words_list['src_words'].append(word)
                        cur_src_words.append(word)
                    else:
                        self.words_list['src_words'].extend(' '.join(word).split())
                        cur_src_words.extend(' '.join(word).split())

            # tokenize for target language
            self.words_list['target_words'].extend(target_sent.split())

            sents_pair = (' '.join(cur_src_words), target_sent+' '+'<end>')

            self.samples.append(sents_pair)


    def __del__(self):
        self.fd_src.close()
        self.fd_target.close()



# step 1.
# parse from dataset
# to make raw corpora pair
# language_anno = './data_eval/Languagemodel_dataset.csv'
# data = pd.read_csv(language_anno, sep='\t', header=None, names=['src','target'])
#
# hanja = []
# ko = []
#
# for s,t in zip(data['src'].values.tolist()[1:], data['target'].values.tolist()[1:]):
#     if not s.strip() or not t.strip():
#         continue
#     hanja.append(s.strip()+'\n')
#     ko.append(t.strip()+'\n')
#
# fp_hanja = open('./data_eval/raw_corpora.hanja','w')
# fp_ko = open('./data_eval/raw_corpora.ko','w')
#
# fp_hanja.writelines(hanja)
# fp_ko.writelines(ko)

# step 2.
# Make chunked data(sample.pkl)
# dataset = Dataset(src_file='./data_eval/raw_corpora.hanja', target_file='./data_eval/raw_corpora.ko')

# step 3.
# split dataset (train/valid/test)
samples = pickle.load(file('data_eval/samples.pkl','rb'))
np.random.seed(5)
np.random.shuffle(samples)

train_samples = samples[:int(0.7*len(samples))]
test_valid_samples = samples[int(0.7*len(samples)):]
test_samples = test_valid_samples[:int(0.5*len(test_valid_samples))]
valid_samples = test_valid_samples[int(0.5*len(test_valid_samples)):]

pickle.dump(train_samples, file('data_eval/train.pkl','wb'))
pickle.dump(test_samples, file('data_eval/test.pkl','wb'))
pickle.dump(valid_samples, file('data_eval/valid.pkl','wb'))