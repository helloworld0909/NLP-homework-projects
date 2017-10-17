import logging
from util.editDistance import editOpts, normalizeOpt
from nltk.probability import FreqDist


class Vocab(object):

    def __init__(self):
        self.token2idx = {'PADDING_TOKEN': 0, 'UNKNOWN_TOKEN': 1}

    def loadVocab(self, vocabFilePath):
        with open(vocabFilePath, 'r', encoding='utf-8') as vocabFile:
            for line in vocabFile:
                token = line.strip()
                self.token2idx[token] = len(self.token2idx)
        logging.info('VocabSize: {}'.format(len(self.token2idx)))

    def inVocab(self, token):
        return token in self.token2idx or token.lower() in self.token2idx

    def getEditDistanceCandidates(self, token, threshold=2):
        candidates = []
        for each in self.token2idx.keys():
            opts = editOpts(each, token, threshold)
            if opts is not None:
                normOpts = list(map(lambda opt: normalizeOpt(each, token, opt), opts))
                candidates.append((each, normOpts))

        return candidates

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
    )


    vocab = Vocab()
    vocab.loadVocab('E:/python_workspace/NLP-homework-projects/hw1/data/vocab.txt')
    print(vocab.getEditDistanceCandidates('helol'))
    print(vocab.inVocab('estimated'))