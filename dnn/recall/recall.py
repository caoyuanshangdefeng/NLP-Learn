#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : recall.py

# 1.搜索句子向量
# 2.返回结果
from dnn.recall.sentence_vectors import Sentence2Vector
from settings import RECALLTOPK, RECALLCLUSTERS
from pprint import pprint


class Recall:
    def __init__(self, by_word=False, method='fasttext'):
        self.by_word = by_word
        self.method = method
        self.sentence_vec = Sentence2Vector(self.method, self.by_word)
        self.vectorizer, self.features_vec, self.lines_cut, self.search_index = self.sentence_vec.built_vectors()
        # print(f"vectorizer : {self.vectorizer}")
        # print(f"features_vec : {self.features_vec}")
        # print(f"lines_cut : {self.lines_cut}")
        # print(f"search_index : {self.search_index}")

    def predict(self, sentence):
        """

        :param sentence:
        :return:

        """
        sentence_cut = [sentence['cut_by_word']] if self.by_word else [sentence['cut']]
        cur_sentence_vector = self.vectorizer.transform(sentence_cut)
        search_results = self.search_index.search(cur_sentence_vector,
                                                  k=RECALLTOPK,
                                                  k_clusters=RECALLCLUSTERS,
                                                  return_distance=True
                                                  )

        search_results_sort = sorted(search_results, key=lambda item: item[0])
        # pprint(search_results_sort)
        filter_results = []
        for result in search_results[0]:
            distance = result[0]
            key = result[1]
            entries = self.sentence_vec.qa_dict[key]['entity']
            if len(set(entries) & set(sentence['entity'])) > 0:
                filter_results.append(key)
        if len(filter_results) < 1:
            return [i[1] for i in search_results[0]]
        else:
            return filter_results
if __name__ == '__main__':
    recall = Recall(by_word=False, method='fasttext')
