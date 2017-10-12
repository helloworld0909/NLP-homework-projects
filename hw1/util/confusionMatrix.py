import re
import logging


class ConfusionMatrix(object):

    def __init__(self):
        self.errorFreqMatrices = {
            'delete': {},
            'replace': {},
            'insert': {},
            'transposition': {},
        }


    def loadErrors(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                line = line.strip('\r\n')
                symbol, count = line.split('\t')
                parseReturn = self.parseError(symbol)
                if parseReturn is not None:
                    optType, charPair = parseReturn
                    self.errorFreqMatrices[optType][charPair] = int(count)
        for key in self.errorFreqMatrices.keys():
            logging.info('{} matrix dim: {}'.format(key, len(self.errorFreqMatrices[key])))


    def parseError(self, symbol):
        left, right = symbol.split('|')
        leftLen = len(left)
        rightLen = len(right)

        if leftLen == 1 and rightLen == 2:
            return 'delete', right[0] + right[1]
        elif leftLen == 2 and rightLen == 1:
            return 'insert', left[0] + left[1]
        elif leftLen == 1 and rightLen == 1:
            return 'replace', right[0] + left[0]
        elif leftLen == 2 and rightLen == 2:
            return 'transposition', left[1] + right[1]
        else:
            logging.warning('Cannot parse error symbol "{}"'.format(symbol))
            return None


    def getErrorCount(self, optType, charPair):
        return self.errorFreqMatrices.get(optType, {}).get(charPair, 0)



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
    )

    confMat = ConfusionMatrix()
    confMat.loadErrors('E:/python_workspace/NLP-homework-projects/hw1/data/count_1edit.txt')
    print(confMat.errorFreqMatrices)
    print(confMat.getErrorCount('replace', 'ae'))