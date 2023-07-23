# -*- coding"utf-8 -*-
# @Auther : zhangxiaoer
# @Time : 2023-03-26 22:11 
# @File : word_sequence.py
# sequence to sequence : many to many的结构




class WordSequence:
    PAD_TAG = "PAD"  # 长度不一样的进行填充
    UNK_TAG = 'UNK'
    SOS_TAG = "SOS"  # start do sequence 句子开始的符号
    EOS_TAG = "EOS"  # 结束符
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {
            self.PAD_TAG: self.PAD,
            self.UNK_TAG: self.UNK,
            self.SOS_TAG: self.SOS,
            self.EOS_TAG: self.EOS,
        }
        for i in range(10):
            self.dict[str(i)] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
        self.count = {}

    def fit(self, sentence):
        """
        传入句子，词频统计
        :param sentence: []
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=5, max_count=None, max_feature=None):
        """
        构造词典
        :param min_count:
        :param max_count:
        :param max_feature:
        :return:
        """
        temp = self.count.copy()
        for key in temp:
            cur_count = self.count.get(key, 0)
            if min_count is not None:
                if cur_count < min_count:
                    del self.count[key]
            if max_count is not None:
                if cur_count > min_count:
                    del self.count[key]
        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), key=lambda x: x[1], resource=True)[:max_feature])
        for key in self.count:
            self.dict[key] = len(self.dict)
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len, add_eos=False):
        """
        把sentence 转化为数字序列
        @param sentence:
        @param max_len:
        @param add_eos: true 输出句子长度为max_len + 1(所有输出 确保句子长度为11);  False  输出句子长度为max_len(所有输入);
        @return:

        """
        import time
        # print('sentence:',sentence,type(sentence))
        # time.sleep(3600)
        sentence=sentence
        if len(sentence) > max_len:  # 句子的长度比max_len长的时候
            sentence = sentence[:max_len]  # 保留的部分
        sentence_len = len(sentence)  # 提前计算句子长度统一
        # print(f"{type(sentence)}--->{sentence_len}--->{sentence}")
        if add_eos:
            sentence += [self.EOS_TAG]

        if sentence_len < max_len:
            sentence = sentence + [self.PAD_TAG] * (max_len - sentence_len)  # 进行填充
        result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def inverse_transform(self, indices):
        """
        把序列转回字符串
        @param indices:
        @return:
        """

        return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices if i != self.EOS]

    def __len__(self):

        return len(self.dict)


if __name__ == "__main__":
    pass



