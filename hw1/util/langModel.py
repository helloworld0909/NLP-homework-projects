import logging
from collections import defaultdict
from nltk.corpus import brown
from nltk import FreqDist


class LangModel(object):

    def __init__(self, nltkCorpus):
        self.corpus = nltkCorpus
        self.models = defaultdict(dict)
        self.vocabSize = 0
        self.wordcount = 0

    def initUnigram(self):
        self.models[1] = FreqDist(self.corpus.words())
        self.vocabSize = len(self.models[1])
        self.wordcount = sum(self.models[1].values())

        logging.info('Unigram vocabSize: {}'.format(self.vocabSize))
        logging.info('Unigram word count: {}'.format(self.wordcount))

    # Smoothing
    def getProb(self, query, ngram=1):
        return (self.models[ngram].get(query, 0) + 1.0) / (self.wordcount + self.vocabSize)

    def getCharCount(self):
        charCount = defaultdict(int)
        for token, freq in self.models[1].items():
            for c in token:
                charCount[c] += freq
        return charCount

    def getCharPairCount(self):
        charPairCount = defaultdict(int)
        for token, freq in self.models[1].items():
            for idx in range(len(token)):
                if idx == 0:
                    charPairCount['>' + token[idx]] += freq
                else:
                    charPairCount[token[idx - 1:idx + 1]] += freq
        return charPairCount


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
    )

    langModel = LangModel(nltkCorpus=brown)
    langModel.initUnigram()
    print(langModel.getProb('the', ngram=1))