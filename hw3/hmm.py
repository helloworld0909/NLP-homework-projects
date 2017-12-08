import os
from collections import defaultdict
import seqlearn.hmm
from seqlearn.datasets import load_conll
from tagger.postagger import Pos

class HMMFactory(object):

    def __init__(self):
        self.pos = Pos()

    def features(self, sentence, i):
        """Features for i'th token in sentence.
                Currently baseline named-entity recognition features, but these can
                easily be changed to do POS tagging or chunking.
                """
        try:
            word, pos = sentence[i]
            yield pos
        except:
            yield sentence[i].split()[-1]


if __name__ == '__main__':

    hmmFactory = HMMFactory()


    with open('crf_input/trigger_train.txt', 'r', encoding='utf-8') as inputFile:
        train = load_conll(inputFile, hmmFactory.features, split=True)
        X_train, y_train, lengths_train = train
    with open('crf_input/trigger_test.txt', 'r', encoding='utf-8') as inputFile:
        test = load_conll(inputFile, hmmFactory.features)
        X_test, y_test, lengths_test = test

    model = seqlearn.hmm.MultinomialHMM(decode='viterbi', alpha=0.01)
    model.fit(X_train, y_train, lengths_train)
    prediction = model.predict(X_test, lengths_test)
    print(list(filter(lambda p: p != 'O', prediction)))








