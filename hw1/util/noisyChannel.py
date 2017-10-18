import logging
import nltk
from util import vocabulary, confusionMatrix, editDistance, langModel

class NoisyChannel(object):

    def __init__(self, inputPath, candidateThreshold=2):
        self.vocab = vocabulary.Vocab()
        self.vocab.loadVocab(inputPath + 'vocab.txt')

        self.confMat = confusionMatrix.ConfusionMatrix()
        self.confMat.loadErrors(inputPath + 'count_1edit.txt')

        self.langModel = langModel.LangModel(nltkCorpus=nltk.corpus.reuters, vocab=self.vocab)
        self.langModel.initUnigram()

        self.charCount = self.langModel.getCharCount()
        self.charPairCount = self.langModel.getCharPairCount()

        self.candidateThreshold = candidateThreshold

    def inVocab(self, token):
        return self.vocab.inVocab(token)

    def getCandidates(self, token, threshold=2):
        return self.vocab.getEditDistanceCandidates(token, threshold)

    def getCandidateProb(self, candidate):
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

            if totalCount == 0:
                return 0.0

            conditionalProb *= float(errorCount) / totalCount

        wordProb = self.langModel.getProb(token, ngram=1)
        prob = wordProb * conditionalProb

        return prob

    def getCorrectToken(self, token):
        candidates = self.getCandidates(token, threshold=self.candidateThreshold)
        if not candidates:
            return token
        else:
            return max(candidates, key=self.getCandidateProb)[0]



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
    )

    model = NoisyChannel(inputPath='E:/python_workspace/NLP-homework-projects/hw1/data/')
    print(model.getCorrectToken('teh'))
    print(model.getCorrectToken('miann'))

