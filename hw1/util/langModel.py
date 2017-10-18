import logging
from collections import defaultdict
from nltk.corpus import brown
from nltk import FreqDist
from util import vocabulary


class LangModel(object):

    def __init__(self, nltkCorpus, vocab):
        self.corpus = nltkCorpus
        self.models = {}
        self.vocabSize = {}
        self.wordcount = {}
        self.nCount = {}
        self.vocab = vocab

    def initUnigram(self):
        words = self.corpus.words()
        self.models[1] = FreqDist(words)
        self.vocabSize[1] = len(self.models[1])
        self.wordcount[1] = sum(self.models[1].values())

        self.nCount[1] = FreqDist(list(self.models[1].values()))
        self.nCount[1][0] = len(tuple(filter(lambda token: self.models[1][token] == 0, self.vocab.token2idx.keys())))
        logging.debug('nCount: {}'.format(dict(self.nCount[1])))

        logging.info('Unigram vocabSize: {}'.format(self.vocabSize[1]))
        logging.info('Unigram word count: {}'.format(self.wordcount[1]))

    def initBigram(self):

        def bigramGenerator(sents):
            for sent in sents:
                for idx in range(len(sent)):
                    if idx == 0:
                        yield ('<s>', sent[idx])
                    else:
                        yield (sent[idx - 1], sent[idx])
        self.initUnigram()
        self.models[2] = FreqDist(bigramGenerator(self.corpus.sents()))
        self.vocabSize[2] = len(self.models[2])
        self.wordcount[2] = sum(self.models[2].values())

        self.nCount[2] = FreqDist(list(self.models[2].values()))
        self.nCount[2][0] = self.nCount[2][1]
        logging.debug('nCount: {}'.format(dict(self.nCount[1])))

        logging.info('Bigram vocabSize: {}'.format(self.vocabSize[2]))
        logging.info('Bigram word count: {}'.format(self.wordcount[2]))


    def getCstar(self, c, cutoff=20, ngram=1):
        if c <= cutoff:
            div = self.nCount[ngram][c]
            if div != 0:
                return (c + 1) * self.nCount[ngram][c + 1] / div
            else:
                return 0
        else:
            return c


    # Good Turing Smoothing
    def getProb(self, query, ngram=1):
        if ngram == 1:
            return self.getCstar(self.models[ngram][query], ngram=ngram) / self.wordcount[ngram]
        elif ngram == 2:
            return self.getCstar(self.models[ngram][query], ngram=ngram) / self.models[ngram - 1][query[0]]

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

    langModel = LangModel(nltkCorpus=brown, vocab=vocabulary.Vocab())
    langModel.initUnigram()
    print(langModel.getProb('the', ngram=1))