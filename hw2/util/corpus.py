import re
import json
import jieba
import logging


class CorpusCN(object):

    token2idx = {'PADDING_TOKEN': 0,'UNKNOWN_TOKEN': 1}
    label2idx = {}

    def __init__(self):
        pass

    def extendToken2idx(self, tokenList):
        newCnt = 0
        for token in tokenList:
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
                newCnt += 1
        return newCnt

    def extendLabel2idx(self, label):
        if label not in self.label2idx:
            self.label2idx[label] = len(self.label2idx)
            logging.debug('New label: {}'.format(label))

    def translateTokenList(self, tokenList):
        return list(map(lambda t: self.token2idx.get(t, 1), tokenList))

    @staticmethod
    def segmentSent(paragraph):
        sents = paragraph.split('\n')
        res = []
        for sent in sents:
            res.extend(re.findall(r"[^。！？]+[。！？]|[^。！？]+$", sent))
        return filter(lambda s: s, res)

    def loadJsonFile(self, filename):
        X_ = {}

        with open(filename, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                vec = []
                jsonObj = json.loads(line)

                titleTokenList = jsonObj['title']
                self.extendToken2idx(titleTokenList)
                vec.append(self.translateTokenList(titleTokenList))

                contentTokenList = jsonObj['content']
                contentVec = self.translateTokenList(contentTokenList)
                vec.append(contentVec)

                X_[jsonObj['id']] = vec

        logging.info('VocabSize: ' + str(self.getVocabSize()))
        return X_

    def loadLabel(self, labelFilename):
        y_ = {}
        relation = []
        with open(labelFilename, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                label, docs = line.strip().split('\t')
                self.extendLabel2idx(label)
                docidList = list(map(int, docs.split(',')))
                relation.append(docidList)
                for docid in docidList:
                    y_[docid] = self.label2idx[label]
        return y_, relation

    def getVocabSize(self):
        return len(self.token2idx)

    def getLabelSize(self):
        return len(self.label2idx)

if __name__ == '__main__':
    corpus = CorpusCN()
    X = corpus.loadJsonFile('../data/news.json')
    y_train = corpus.loadLabel('../data/train.txt')
    y_test = corpus.loadLabel('../data/test.txt')

    print(corpus.getVocabSize())
    print(X[30819])