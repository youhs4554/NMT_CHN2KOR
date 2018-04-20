#-*- coding: utf-8 -*-

import sys
sys.path.insert(0, './coco-caption') # Hack to allow the import of pycocoeval

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

class NMTEval:
    def __init__(self,sentences,gts,res):
        self.evalSents = []
        self.eval = {}
        self.sentToEval = {}
        self.params = {'sentence_id': sentences}
        self.gts = gts
        self.res = res

    def evaluate(self):
        sentIds = self.params['sentence_id']
        gts = self.gts
        res = self.res

        # =================================================
        # Set up scorers
        # =================================================
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setSentToEval(scs, sentIds, m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.setSentToEval(scores, sentIds, method)
                print "%s: %0.3f"%(method, score)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setSentToEval(self, scores, sentIds, method):
        for sentId, score in zip(sentIds, scores):
            if not sentId in self.sentToEval:
                self.sentToEval[sentId] = {}
                self.sentToEval[sentId]["sentence_id"] = sentId
            self.sentToEval[sentId][method] = score

    def setEvalImgs(self):
        self.evalSents = [eval for imgId, eval in self.sentToEval.items()]

def calculate_metrics(rng,datasetGTS,datasetRES):
    sentIds = rng
    gts = {}
    res = {}

    src2tgtGTS = {ann['sentence_id']: [] for ann in datasetGTS['annotations']}
    for ann in datasetGTS['annotations']:
        src2tgtGTS[ann['sentence_id']] += [ann]

    src2tgtRES = {ann['sentence_id']: [] for ann in datasetRES['annotations']}
    for ann in datasetRES['annotations']:
        src2tgtRES[ann['sentence_id']] += [ann]

    for sentId in sentIds:
        gts[sentId] = src2tgtGTS[sentId]
        res[sentId] = src2tgtRES[sentId]

    evalObj = NMTEval(sentIds,gts,res)
    evalObj.evaluate()
    return evalObj.eval

if __name__ == '__main__':
    rng = range(2)
    datasetGTS = {
        'annotations': [{u'sentence_id': 0, u'caption': u'춘당대 에 거둥 하여 경모궁 에 상 이 행차 하 었 을 때 수행 하 ㄴ 장신 과 무사 등 의 시사 를 행하 였 다'},
                        {u'sentence_id': 1, u'caption': u'사복시 가 아뢰 었 다 팔 도 의 각 목장 에서 금년 현재 방목 하 고 있 는 암수 의 말 이 모두 6 천 8 백 76 필 이 ㅂ니다'}]
        }
    datasetRES = {
        'annotations': [{u'sentence_id': 0, u'caption': u'경모궁 에 상 이 행차 하 었 을 때 수행 하 ㄴ 노비 와 무사 등 의 시사 를 행하 였 다'},
                        {u'sentence_id': 1, u'caption': u'사복시 가 아뢰 었 다 팔 도 의 각 목장 에서 현재 방목 중인 암수 의 소 는 모두 6 천 필 이 ㅂ니다'}]
        }
    print calculate_metrics(rng,datasetGTS,datasetRES)