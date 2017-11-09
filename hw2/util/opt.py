import numpy as np

def transform(X_all, y_dict, callback=lambda x: x[1]):
    lookup = {}
    X_transform = []
    y_transform = []
    for docid, labelIdx in y_dict.items():
        lookup[docid] = len(lookup)
        X_transform.append(callback(X_all[docid]))
        y_transform.append(labelIdx)
    return X_transform, y_transform, lookup

def predictionFusion(prediction, relation, lookup, idx2label):
    with open('result.txt', 'w', encoding='utf-8') as outputFile:
        for docidList in relation:
            mergeLogLikelihood = np.zeros(len(idx2label))
            for docid in docidList:
                idx = lookup[docid]
                mergeLogLikelihood += prediction[idx]
            chosenLabel = idx2label[mergeLogLikelihood.argmax()]
            outputFile.write(chosenLabel + '\n')