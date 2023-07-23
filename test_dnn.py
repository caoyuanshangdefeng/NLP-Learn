#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
from dnn.sort.word_sequence import WordSequence
from dnn.recall.recall import Recall
from dnn.recall.fasttext_vectorizer import build_model
from settings import SORTRESP, SORTINPUTDATA
from dnn.sort.dataset import data_loader
from dnn.sort.train import run
sort_ws_path = r".sort_ws.pkl"
ws = pickle.load(open(sort_ws_path, 'rb'))
if __name__ == '__main__':
    recall = Recall(by_word=False,method='fasttext')
    # sentence = {
    #     'cut_by_word': 'java 是 什 么',
    #     'cut': "java 是 什么",
    #     'entity':['java']
    # }
    # resp=recall.predict(sentence)
    # print(resp)
    # build_model()
    # sort_model_resp = os.path.join(SORTRESP, f"sort_ws.pkl")
    # sort_ws=pickle.load(open(sort_model_resp,'rb'))
    # print(sort_ws)
    # print(sort_ws.inverse_dict)
    # for q,sim_q,q_length,sim_q_length in data_loader:
    #     print(f"q content : {q}")
    #     print(f"sim_q content : {sim_q}")
    #     print(f"q_length content : {q_length}")
    #     print(f"sim_q_length content : {sim_q_length}")
    #     break
    run()

