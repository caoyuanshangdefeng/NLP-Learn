#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/25 22:13
# @Author  : woodcutter
# @File    : sk_case.py
# https://www.jianshu.com/p/622222b96f76
import os
import json
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# from data_utils import *
import jieba
import matplotlib.pyplot as plt
test_file_path = r"E:\learn\cnews\cnews.test.txt"
sk_test_file_path = r"E:\learn\AI\sk_pro\sk_case\sk_test.txt"
def create_test_text():

    test_file_content = open(r"E:\learn\cnews\cnews.test.txt", 'r', encoding='utf-8').readlines()
    theme_content_map = {}
    for row_item in test_file_content:
        row_item_parse = row_item.split('\t')
        theme = row_item_parse[0]
        content = row_item_parse[1]
        if not theme_content_map.get(theme, None):
            theme_content_map[theme] = [content]
        else:
            theme_content_map[theme].append(content)
    print(len(theme_content_map))
    test_text_list = []
    count = 0
    for theme in theme_content_map:
        if count >= 5:
            break
        content_list = theme_content_map[theme]
        random_list = random.sample(content_list, 33)
        test_text_list.extend(random_list)
        count += 1
    random.shuffle(test_text_list)
    with open(sk_test_file_path, 'w', encoding='utf-8') as fw:
        fw.write(''.join(test_text_list))



# bigram分词
segment_bigram = lambda text: " ".join([word + text[idx + 1] for idx, word in enumerate(text) if idx < len(text) - 1])
# 结巴中文分词
segment_jieba = lambda text: " ".join(jieba.cut(text))

'''
1、加载语料
'''
corpus = []
with open(sk_test_file_path, "r", encoding="utf-8") as f:
    for line in f:
        # 去掉标点符号
        corpus.append(segment_jieba(line.strip()))


# 2.计算tf-idf设为权重

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

'''
    3、获取词袋模型中的所有词语特征
    如果特征数量非常多的情况下可以按照权重降维
'''

word = vectorizer.get_feature_names_out()
print("word feature length: {}".format(len(word)))

'''
    4、导出权重，到这边就实现了将文字向量化的过程，矩阵中的每一行就是一个文档的向量表示
'''
tfidf_weight = tfidf.toarray()


'''
    5、对向量进行聚类
'''

# 指定分成7个类
kmeans = KMeans(n_clusters=5)
kmeans.fit(tfidf_weight)

# 打印出各个族的中心点
print(kmeans.cluster_centers_)
for index, label in enumerate(kmeans.labels_, 1):
    print("index: {}, label: {}".format(index, label))

# 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
# k-means的超参数n_clusters可以通过该值来评估
print("inertia: {}".format(kmeans.inertia_))
'''
    6、可视化
'''

# 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
tsne = TSNE(n_components=2)
decomposition_data = tsne.fit_transform(tfidf_weight)

x = []
y = []

for i in decomposition_data:
    x.append(i[0])
    y.append(i[1])

fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
plt.scatter(x, y, c=kmeans.labels_, marker="x")
plt.xticks(())
plt.yticks(())
plt.show()
# plt.savefig('./sample.png', aspect=1)