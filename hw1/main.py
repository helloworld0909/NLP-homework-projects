import logging
import nltk
from util.noisyChannel import NoisyChannel

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

inputPath = 'E:/python_workspace/NLP-homework-projects/hw1/data/'

model = NoisyChannel(inputPath, candidateThreshold=2)

with open(inputPath + 'testdata.txt', 'r', encoding='utf-8') as inputFile:
    with open(inputPath + 'result.txt', 'w', encoding='utf-8') as outputFile:

        correctionCount = 0
        for line in inputFile:
            line = line.strip('\r\n').split('\t')
            sentID = line[0]
            sent = line[2]

            tokenList = nltk.word_tokenize(sent)
            for idx, token in enumerate(tokenList):
                if not model.inVocab(token):
                    correctToken = model.getCorrectToken(token)
                    tokenList[idx] = correctToken
                    if token != correctToken:
                        correctionCount += 1
                        logging.debug(token + ' -> ' + correctToken)
            correctSent = ' '.join(tokenList)
            outputFile.write(sentID + '\t' + correctSent + '\n')
        logging.info('Correction count: {}'.format(correctionCount))
