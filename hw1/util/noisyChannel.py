import logging
import math
import nltk
from util import vocabulary, confusionMatrix, editDistance, langModel

class NoisyChannel(object):

    def __init__(self, inputPath, candidateThreshold=2, ngram=2):
        self.vocab = vocabulary.Vocab()
        self.vocab.loadVocab(inputPath + 'vocab.txt')

        self.confMat = confusionMatrix.ConfusionMatrix()
        self.confMat.loadErrors(inputPath + 'count_1edit.txt')

        self.langModel = langModel.LangModel(nltkCorpus=nltk.corpus.reuters, vocab=self.vocab)
        if ngram == 1:
            self.langModel.initUnigram()
        elif ngram == 2:
            self.langModel.initBigram()
        else:
            raise NotImplementedError('Not support this n-gram model')

        self.charCount = self.langModel.getCharCount()
        self.charPairCount = self.langModel.getCharPairCount()

        self.candidateThreshold = candidateThreshold

    def inVocab(self, token):
        return self.vocab.inVocab(token)

    def getCandidates(self, token, threshold=2):
        return self.vocab.getEditDistanceCandidates(token, threshold)

    def getCandidateProb(self, candidate, before):
        token, opts = candidate
        conditionalProb = 1.0
        for opt in opts:
            optType, charPair = opt
            errorCount = self.confMat.getErrorCount(optType, charPair)
            if optType == 'insert':
                totalCount = self.charCount[charPair[0]]
            elif optType == 'replace':
                totalCount = self.charCount[charPair[1]]
            else:
                totalCount = self.charPairCount[charPair]

            conditionalProb *= (float(errorCount) + 1) / (totalCount + self.confMat.getErrorVocabSize(optType))

        ngram = len(before) + 1
        query = tuple(before + [token])
        wordProb = self.langModel.getProb(query, ngram=ngram)
        prob = wordProb * conditionalProb

        return prob

    def getCorrectToken(self, token, before):
        candidates = self.getCandidates(token, threshold=self.candidateThreshold)
        if not candidates:
            return token
        else:
            return max(candidates, key=lambda candidate: self.getCandidateProb(candidate, before))[0]

    def generateBefore(self, tokenList, tokenIdx, ngram):
        if ngram == 1:
            return []
        elif ngram == 2:
            if tokenIdx == 0:
                before = ['<s>']
            else:
                before = [tokenList[tokenIdx - 1]]
        else:
            raise NotImplementedError('Not support this n-gram model')
        return before



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
    )

    model = NoisyChannel(inputPath='E:/python_workspace/NLP-homework-projects/hw1/data/')
    print(model.getCorrectToken('Teh', ('<s>', )))
    print(model.getCorrectToken('miann', ('the', )))

