#-*- coding: utf-8 -*-
import numpy as np
import glob
import json
import os
from konlpy.tag import Kkma
from collections import Counter
import re
import tqdm
from natsort import natsorted
import codecs

def merge_and_encode(root_dir, fname):
    # merge all txt files(to convert it to utf-8 at once)
    files = natsorted(glob.glob(os.path.join(root_dir, '*/*/*')))

    with open(fname,'w') as f:
        for each in files:
            with codecs.open(each,'r',encoding='utf-8-sig') as g:
                for line in g:
                    clean_line = line.strip()
                    if clean_line:
                        f.write(clean_line.encode('utf-8')+'\n')

    # change encoding utf-16 to utf-8
    # os.system('iconv -f UTF-16 -t UTF-8 tmp.txt -o raw_utf8.txt && rm tmp.txt')

def preprocess_rawcorpus(raw_corpus_path, kor_file_path, hanja_file_path, sep):
    with open(raw_corpus_path) as f:
        data = f.read().decode('utf-8')
    os.remove(raw_corpus_path)

    f_kor = file(kor_file_path, mode='w')
    f_hanja = file(hanja_file_path, mode='w')

    for each in data.split(sep):
        try:
            kor = ' '.join([ kk for kk in each.split(u'국역')[1].split(u'원문')[0].split() if not kk.startswith(u'[') ])
            if u'원문에는' not in each:
                hanja = ' '.join(each.split(u'원문')[1].split())[1:]
        except:
            continue
        f_kor.write(kor.encode('utf-8')+'\n')
        f_hanja.write(hanja.encode('utf-8')+'\n')

    f_kor.close()
    f_hanja.close()

def basic_tokenizer(sent):
    # Regular expressions used to tokenize.
    _WORD_SPLIT = re.compile(b"([·.,!?\"':;)(])")

    words = []
    for word in sent.strip().split():
        words.extend(_WORD_SPLIT.split(word))

    return [w for w in words if w]

class Dataset(object):
    def __init__(self, src_file, target_file, vocab_size, train_ratio=0.7, test_ratio=0.3):
        self.create_vocab(src_file, target_file, vocab_size)
        # convert word to ix
        for n, (src_sent, target_sent) in enumerate(self.samples):
            src_ixs = []
            target_ixs = []
            for w in src_sent.split():
                w = self.word2ix['src'].get(w,'<unk>')
                src_ixs.append(self.word2ix['src'][w])

            for w in target_sent.split():
                w = self.word2ix['target'].get(w,'<unk>')
                target_ixs.append(self.word2ix['target'][w])

            self.samples[n] = dict(src_sent=src_sent,src_ixs=src_ixs, target_sent=target_sent, target_ixs=target_ixs)

        self.n = len(self.samples)
        self.train = self.samples[:int(self.n * train_ratio)]
        self.valid = self.samples[len(self.train):len(self.train) + int(self.n * (test_ratio * 2. / 3))]
        self.test = self.samples[len(self.train) + len(self.valid):]

        # save datasets
        json.dump(self.train, file('./data/train.json', 'wb'))
        json.dump(self.valid, file('./data/valid.json', 'wb'))
        json.dump(self.test, file('./data/test.json', 'wb'))

        # save word2ix
        json.dump(self.word2ix, file('./data/word2ix.json', 'wb'))

    def create_vocab(self, src_file, target_file, vocab_size):
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

            # preprocess target_sent(remove comments and any other noise)
            target_sent = re.sub(r'\d*].*', '', target_sent)
            target_sent = re.sub(r'\([^)]*\)', '', target_sent)
            target_sent = re.sub(r'\d*\)', '', target_sent)

            # src/target pos-tags
            src_tags = kkm.pos(src_sent)
            target_tags = kkm.pos(target_sent)
            
            # extract named_entities for source language
            named_entities = [tag[0] for tag in target_tags if
                              tag[1] == 'OH']  # extract named-entities from target sentence

            # remove all symbols
            src_sent = ' '.join([tag[0] for tag in src_tags if not tag[1].startswith('S')])

            target_sent = ' '.join([tag[0] for tag in target_tags if not tag[1].startswith('S')])

            for ne in named_entities:
                src_sent = src_sent.replace(ne, '*' + ne + '*')
                # target_sent = target_sent.replace('( '+ne, '( '+ne+' )')
                target_sent = target_sent.replace(ne, '') # remove hanja

            src_sent = ' '.join(src_sent.split('*'))

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

        self.word2ix = dict(src=None, target=None)

        src_words_sorted = [w for w, c in
                            sorted(Counter(self.words_list['src_words']).items(), key=lambda x: x[1], reverse=True)][:vocab_size]
        target_words_sorted = [w for w, c in
                               sorted(Counter(self.words_list['target_words']).items(), key=lambda x: x[1],
                                      reverse=True)][:vocab_size]

        self.word2ix['src'] = {'<pad>': 0, '<unk>': 1}  # pre-defined tokens
        self.word2ix['target'] = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}  # pre-defined tokens

        self.word2ix['src'].update(dict(zip(src_words_sorted, range(2, vocab_size))))
        self.word2ix['target'].update(dict(zip(target_words_sorted, range(4, vocab_size))))

    def __del__(self):
        self.fd_src.close()
        self.fd_target.close()

class Batcher(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_generator = self.batch_generator()
        self.epoch = 0

    def batch_generator(self):
        start = 0
        while True:
            mini_batch = dict(src_sent=[], src_ixs=[], target_sent=[], target_ixs=[])

            if start > len(self.dataset) - self.batch_size:
                ixs = np.arange(len(self.dataset))
                np.random.shuffle(ixs)
                self.dataset = list(np.array(self.dataset)[ixs])
                self.epoch += 1
                start = 0
                print 'end of batch, shuffle...'

            cur_batch = self.dataset[start:start + self.batch_size]

            maxlen_src = max([len(sample['src_ixs']) for sample in cur_batch])
            maxlen_target = max([len(sample['target_ixs']) for sample in cur_batch])

            # padding with zero
            for sample in cur_batch:
                src_ix_padded = sample['src_ixs'] + [0] * (maxlen_src - len(sample['src_ixs']))
                target_ix_padded = sample['target_ixs'] + [0] * (maxlen_target - len(sample['target_ixs']))
                mini_batch['src_sent'].append(sample['src_sent'])
                mini_batch['target_sent'].append(sample['target_sent'])
                mini_batch['src_ixs'].append(src_ix_padded)
                mini_batch['target_ixs'].append(target_ix_padded)

            start += self.batch_size

            yield mini_batch

    def next_batch(self):
        return self.batch_generator.next()


if __name__=='__main__':
    pass
    # step 1.
    # merge_and_encode(root_dir='../NMT_data/chosun-form1',
    #                  fname='./data/raw_utf-8.txt')
    # merge_and_encode(root_dir='../NMT_data/chosun-form2',
    #                  fname='./data/raw_utf-8-2.txt')

    # step 2. make each corpora text file
    # preprocess_rawcorpus(raw_corpus_path='./data/raw_utf-8.txt',
    #                      kor_file_path='./data/raw_corpora.ko',
    #                      hanja_file_path='./data/raw_corpora.hanja', sep=u'【태백산사고본】')
    # preprocess_rawcorpus(raw_corpus_path='./data/raw_utf-8-2.txt',
    #                      kor_file_path='./data/raw_corpora-2.ko',
    #                      hanja_file_path='./data/raw_corpora-2.hanja', sep=u'【원본】')
    # concat all text file as one file
    # for k,h in zip(glob.glob('./data/raw_corpora-*.ko'), glob.glob('./data/raw_corpora-*.hanja')):
    #     os.system('cat {}>>./data/raw_corpora.ko'.format(k))
    #     os.system('cat {}>>./data/raw_corpora.hanja'.format(h))
    #     os.remove(k); os.remove(h)

    # step 3. preprocess & save dataset
    # dataset = Dataset(src_file='./data/raw_corpora.hanja', target_file='./data/raw_corpora.ko', vocab_size=40000)