import logging
from util.corpus import CorpusCN
from util import opt
from models import nb

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

corpus = CorpusCN()
X = corpus.loadJsonFile('data/news.json')
y_trainDict, relation_train = corpus.loadLabel('data/train.txt')
y_testDict, relation_test = corpus.loadLabel('data/test.txt')

model = nb.NaiveBayes(vocabSize=corpus.getVocabSize(), labelSize=corpus.getLabelSize())
callback = lambda x: x[0]
X_train, y_train, lookup_train = opt.transform(X, y_trainDict, callback)
X_test, _, lookup_test = opt.transform(X, y_testDict, callback)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

idx2label = {v: k for k, v in corpus.label2idx.items()}
opt.predictionFusion(y_predict, relation_test, lookup_test, idx2label)
