#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""recall"""
import json
import os
from settings import QACORPUSPATH, search_index_bm_25_by_word,search_index_bm_25,search_index_fasttext_by_word,search_index_fasttext,search_index_by_word,search_index
from sklearn.feature_extraction.text import TfidfVectorizer
from dnn.recall.BM25vectorizer import BM25Vectorizer
import pysparnn.cluster_index as ci
import pickle
from dnn.recall.fasttext_vectorizer import FastTextVectorizer

"""
BM25 句子比较长的情况下，可以抑制词频的数量，机器学习方法，考虑的是单个字，没有考虑连续的多个字，词频角度出发，词的数量出发，词袋模型的变形
没有考虑到语义

"""

class Sentence2Vector:
    def __init__(self, method='BM25', by_word=False):
        self.by_word = by_word
        self.qa_dict = json.load(open(QACORPUSPATH, 'r', encoding='utf-8'))
        if method.lower() == 'bm25':
            self.vectorizer = BM25Vectorizer()
            self.search_index_path = search_index_bm_25_by_word if self.by_word else search_index_bm_25
        if method.lower() == 'fasttext':
            self.vectorizer = FastTextVectorizer()
            self.search_index_path = search_index_fasttext_by_word if self.by_word else search_index_fasttext
        else:
            self.vectorizer = TfidfVectorizer()
            self.search_index_path = search_index_by_word if self.by_word else search_index

    def built_vectors(self):

        lines = list(self.qa_dict.keys())
        lines_cut = [' '.join(self.qa_dict[q]['q_cut_by_word']) for q in lines] if self.by_word else [
            ' '.join(self.qa_dict[q]['q_cut']) for q in lines]
        # print(f'lines_cut', len(lines_cut), type(lines_cut),lines_cut)
        tfidf_vectorizer = self.vectorizer
        features_vec = tfidf_vectorizer.fit_transform(lines_cut)
        # print(f'features_vec', features_vec)
        search_index = self.get_cp(features_vec, lines)
        return self.vectorizer, features_vec, lines_cut, search_index

    def get_cp(self, vectors, data):
        if os.path.exists(self.search_index_path):
            search_index = pickle.load(open(self.search_index_path, 'rb'))
        else:
            search_index = self.build_cp(vectors, data)
        return search_index

    def build_cp(self, vectors, data):
        # 构造索引
        search_index = ci.MultiClusterIndex(vectors, data)
        pickle.dump(search_index, open(self.search_index_path, 'wb'))
        return search_index
        # # 对用户输入的句子进行向量化
        # tfidf_vec = 1
        # ret = None
        # search_vector = tfidf_vec.fit_transform(ret)
        # # 搜索获取结果，返回最大的8个数据，之后根据"main_entity"进行结果过滤
        # cp_search_list = search_index.search(search_vector,
        #                                      k=8,
        #                                      k_clusters=10,
        #                                      return_distance=True
        #                                      )
        # exists_same_entity = False
        # search_list = []
        # main_entity = []
        # for _temp_call_line in cp_search_list[0]:
        #     cur_entity = self.qa_dict[_temp_call_line[1]]["main_entity"]
        #     # 命名体的集合存在交集时返回
        #     if len(set(main_entity) & set(cur_entity)) > 0:
        #         exists_same_entity = True
        #         search_list.append(_temp_call_line[1])
        #
        # if exists_same_entity:  # 存在相同主体的时候
        #     return search_list
        # else:
        #     return [_temp_call_line[1] for _temp_call_line in cp_search_list]


if __name__ == '__main__':
    s2v = Sentence2Vector()
    test_resp = s2v.built_vectors()
