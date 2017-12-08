import os
import sys
from tagger.postagger import Pos
from postprocess import Postprocess

def enrichCoNLL(filename):
    postagger = Pos()

    with open('input/' + filename, encoding='utf-8') as inputFile:
        with open('crf_input/' + filename, 'w', encoding='utf-8') as outputFile:
            sentence = []
            tags = []
            for line in inputFile:
                line = line.strip()
                if not line:

                    posSeq = postagger.postag(sentence)
                    for outputLine in zip(sentence, posSeq, tags):
                        outputFile.write('\t'.join(outputLine) + '\n')
                    outputFile.write('\n')

                    sentence = []
                    tags = []
                    continue
                else:
                    token, label = line.split('\t')
                    sentence.append(token)
                    tags.append(label)

if __name__ == '__main__':
    crf_path = 'E:/data/CRF++-0.54/'
    crf_data_path = 'crf_input/'

    para = 'trigger'

    os.system('{}crf_learn -a MIRA -f 5 {}template.txt {}{para}_train.txt {para}_model'.format(crf_path, crf_data_path, crf_data_path, para=para))

    if para == 'trigger':
        os.system('{}crf_test -m model -v2 {}{para}_test.txt > {para}_raw.txt'.format(crf_path, crf_data_path, para=para))
        Postprocess().processOutput('trigger_raw.txt', 'trigger_result.txt')
    else:
        os.system('{}crf_test -m {para}_model {}{para}_test.txt > {para}_result.txt'.format(crf_path, crf_data_path, para=para))
