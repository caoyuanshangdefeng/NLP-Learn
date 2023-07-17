#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/3 23:45
# @Author  : 草原上的风
# @File    : test_BM25.py
from dnn.recall.BM25vectorizer import BM25Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
bm25_vec=BM25Vectorizer()
data=[
    'hello hello world',
    'oh hello hello',
    'Play it',
    'Play it again Sam,24343,123'
]
bm25_resp=bm25_vec.fit_transform(data)
print(f"bm25_resp : {bm25_resp}")
tfidf_vec=TfidfVectorizer()
tfidf_resp=tfidf_vec.fit_transform(data)
print(f"tfidf_resp : {tfidf_resp.toarray()}")
