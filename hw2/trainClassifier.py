import logging
from util.corpus import CorpusCN
from util import opt
from models import nb
import nltk

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

corpus = CorpusCN()
X = corpus.loadJsonFile('data/news.json', translate=False)
vocabulary = list(map(lambda tf: tf[0], corpus.tokenFreqDist.most_common(1000)))
y_trainDict, relation_train = corpus.loadLabel('data/train.txt')
y_testDict, relation_test = corpus.loadLabel('data/test.txt')
callback = lambda x: x[0]
X_train, y_train, lookup_train = opt.transform(X, y_trainDict, callback)
X_test, y_test, lookup_test = opt.transform(X, y_testDict, callback)

def document_features(x_vec, vocab):
    features = {}
    tokenSet = set(x_vec)
    for token in vocab:
        features[token] = token in tokenSet
    return features

features_train = [(document_features(x_vec, vocabulary), label) for x_vec, label in zip(X_train, y_train)]
features_test = [(document_features(x_vec, vocabulary), label) for x_vec, label in zip(X_test, y_test)]

def naiveBayes():

    logging.info('Begin training...')
    classifier = nltk.NaiveBayesClassifier.train(features_train)
    logging.info('Begin prediction...')
    print('acc:', nltk.classify.accuracy(classifier,features_test))

def decisionTree():
    logging.info('Begin training...')
    classifier = nltk.DecisionTreeClassifier.train(features_train)
    logging.info('Begin prediction...')
    print('acc:', nltk.classify.accuracy(classifier, features_test))

def maxent():
    logging.info('Begin training...')
    classifier = nltk.MaxentClassifier.train(features_train)
    logging.info('Begin prediction...')
    print('acc:', nltk.classify.accuracy(classifier, features_test))


naiveBayes()
decisionTree()
