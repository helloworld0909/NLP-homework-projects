import numpy as np

class NaiveBayes(object):

    def __init__(self, vocabSize, labelSize):
        self.vocabSize = vocabSize
        self.labelSize = labelSize
        self.probMatrix = np.zeros(shape=(labelSize, vocabSize), dtype='float')
        self.probLabel = np.zeros(labelSize, dtype='float')

    @staticmethod
    def freqVecEncode(x_vec, vocabSize):
        freqVec = np.zeros(vocabSize, dtype='float')
        for xIdx in x_vec:
            freqVec[xIdx] += 1
        return freqVec

    def fit(self, X, y):

        # Count
        for x_vec, labelIdx in zip(X, y):
            self.probMatrix[labelIdx] += self.freqVecEncode(x_vec, self.vocabSize)
            self.probLabel[labelIdx] += 1

        # Normalize
        for labelIdx in range(self.probMatrix.shape[0]):
            self.probMatrix[labelIdx] += 1
            self.probMatrix[labelIdx] /= (self.probMatrix[labelIdx].sum() + self.vocabSize)
        self.probLabel /= self.probLabel.sum()


    def predict(self, X):
        llVec = np.zeros((len(X), self.labelSize), dtype='float64')

        for idx, x_vec in enumerate(X):
            for labelIdx in range(self.labelSize):
                llVec[idx][labelIdx] += (np.vectorize(lambda x: np.log(self.probMatrix[labelIdx][x]))(x_vec)).sum()
                llVec[idx][labelIdx] += np.log(self.probLabel[labelIdx])

        return llVec
