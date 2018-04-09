#-*- coding: utf-8 -*-
from numpy.core.defchararray import startswith
import numpy as np
import glob
import os, sys
import pickle, pandas

from konlpy.tag import Kkma, Komoran, Hannanum, Twitter, Mecab
from konlpy.utils import pprint
from collections import Counter
import re

def merge_and_encode(root_dir):
    # merge all txt files(to convert it to utf-8 at once)
    files = glob.glob(os.path.join(root_dir, '*/*/*'))

    with open('raw_utf-8.txt','w') as f:
        for each in files:
            with open(each) as g:
                for line in g:
                    f.write(line)

    # change encoding utf-16 to utf-8
    # os.system('iconv -f UTF-16 -t UTF-8 tmp.txt -o raw_utf8.txt && rm tmp.txt')

# uncomment for new data
merge_and_encode('chosun_full')

with open('raw_utf8.txt') as f:
    data = f.read().decode('utf-8')

f_kor = file('raw_corpora.ko', mode='w')
f_hanja = file('raw_corpora.hanja', mode='w')

for each in data.split(u'【태백산사고본】'):
    try:
        kor = ' '.join([ kk for kk in each.split(u'국역')[1].split(u'원문')[0].split() if not kk.startswith(u'[') ])
        hanja = ' '.join(each.split(u'원문')[1].split())[1:]
    except:
        continue
    f_kor.write(kor.encode('utf-8')+'\n')
    f_hanja.write(hanja.encode('utf-8')+'\n')

f_kor.close()
f_hanja.close()