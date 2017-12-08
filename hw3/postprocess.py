import numpy as np

class Postprocess(object):

    def processOutput(self, filename, outputfilename):
        with open(outputfilename, 'w', encoding='utf-8') as outputFile:
            with open(filename, 'r', encoding='utf-8') as inputFile:
                lines = []
                for line in inputFile:
                    line = line.strip()
                    if not line:
                        positiveLabelCount = self.countPositiveLabel(lines)
                        if positiveLabelCount != 1:

                            probDistSeq, labelList = self.parseLines(lines)
                            labelPosition, label = self.predictWithConstraint(probDistSeq, labelList)
                            for idx, rawLine in enumerate(lines):
                                dataList = list(filter(lambda token: '/' not in token, rawLine.split('\t')))

                                if idx == labelPosition:
                                    outputFile.write(dataList[0] + '\t' + dataList[-1] + '\t' + label + '\n')
                                else:
                                    outputFile.write(dataList[0] + '\t' + dataList[-1] + '\tO\n')

                        else:
                            for rawLine in lines:
                                dataList = list(filter(lambda token: '/' not in token, rawLine.split('\t')))
                                predictLabel = list(filter(lambda token: '/' in token, rawLine.split('\t')))[0]
                                predictLabel = predictLabel.split('/')[0]
                                outputFile.write(dataList[0] + '\t' + dataList[-1] + '\t' + predictLabel +'\n')

                        outputFile.write('\n')
                        lines = []

                    elif line.startswith('#'):
                        continue
                    else:
                        lines.append(line)

    @staticmethod
    def parseLines(lines):
        probDistSeq = []
        labelList = []
        for line in lines:
            dataList = line.split('\t')
            dataList = list(filter(lambda token: '/' in token, dataList))
            probDist = map(lambda pair: pair.split('/'), dataList[1:])
            probDist = list(filter(lambda pair: pair[0] != 'O', probDist))
            if not labelList:
                labelList = list(map(lambda pair: pair[0], probDist))
            probDist = list(map(lambda pair: float(pair[1]), probDist))
            probDistSeq.append(probDist)
        return np.array(probDistSeq, dtype='float32'), labelList

    @staticmethod
    def countPositiveLabel(lines):
        count = 0
        for line in lines:
            dataList = line.split('\t')
            dataList = list(filter(lambda token: '/' in token, dataList))
            if dataList[0].split('/')[0] != 'O':
                count += 1
        return count

    @staticmethod
    def predictWithConstraint(probDistSeq, labelList):
        x = probDistSeq.max(axis=-1).argmax()
        y = probDistSeq[x].argmax(axis=-1)
        return x, labelList[y]



if __name__ == '__main__':
    Postprocess().processOutput('trigger_raw.txt', 'trigger_result.txt')




