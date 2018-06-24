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
    import json
    datasetGTS = json.load(file('./results/gold_val.json', 'rb'))
    datasetRES = json.load(file('./results/hypo_val.json', 'rb'))
    rng = range(len(datasetGTS['annotations']))

    print calculate_metrics(rng,datasetGTS,datasetRES)