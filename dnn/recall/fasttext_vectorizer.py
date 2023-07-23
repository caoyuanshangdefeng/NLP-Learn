#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 使用fasttext获取词向量
from fasttext import FastText
import os
import numpy as np
from settings import sim_q_cut_words, fasttext_by_word_model, sim_q_cut, fasttext_model

"""
fit /fɪt/ 适合，合身；匹配，相配；
transform /trænsˈfɔːm/ v.使改观，使变形，使转化；变换
"""

def build_model(by_word=True):
    if by_word:
        data_path = sim_q_cut_words
        save_res_path = fasttext_by_word_model
    else:
        data_path = sim_q_cut
        save_res_path = fasttext_model
    model = FastText.train_unsupervised(data_path, wordNgrams=5, epoch=20, minCount=5, ws=4, lr=0.01)
    model.save_model(save_res_path)


def get_model(by_word=True):
    if by_word:
        save_res_path = fasttext_by_word_model
        return FastText.load_model(save_res_path)
    else:
        raise NotImplementedError


class FastTextVectorizer():
    def __init__(self, by_word=True):
        self.model = get_model(by_word)

    def transform(self, sentences):
        """
        :param sentences: [sentence1,sentence2,sentence3]
        :return:
        """
        results = [self.model.get_sentence_vector(sentence_item) for sentence_item in sentences]
        # print(f" FastTextVectorizer transform sample--->{results[0]}")
        return np.array(results)

    def fit_transform(self, sentences):
        return self.transform(sentences)
