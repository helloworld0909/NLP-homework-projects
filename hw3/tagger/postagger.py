import os
from pyltp import Postagger

class Pos(object):
    LTP_DATA_DIR = 'E:/data/ltp_data_v3.4.0'  # ltp模型目录的路径
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`


    def __init__(self):
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(self.pos_model_path)  # 加载模型

    def postag(self, sentence):
        if not isinstance(sentence, list):
            sentence = list(sentence)
        return self.postagger.postag(list(sentence))

    def release(self):
        self.postagger.release()