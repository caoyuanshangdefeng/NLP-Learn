#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
排序代码的封装
"""
import pickle
import pickle

from tqdm import tqdm
from torch.optim import Adam
from dnn.sort.word_sequence import WordSequence
from settings import QACORPUSPATH, DEVICE, sort_model_save_path, sort_ws, sim_q_cut_words, q_cuted_words
import json
import torch
import torch.nn.functional as F
from train import train_run
from dnn.sort.siamese import SiameseNetwork
from dnn.recall.recall import Recall

recall = Recall(by_word=False, method='fasttext')
# res=recall.sentence_vec.built_vectors()
from utils.cut import cut

model = SiameseNetwork().to(DEVICE)
optimizer = Adam(model.parameters(), lr=0.001)

max_len_flag = 20
ws = pickle.load(open(sort_ws, 'rb'))


class DnnSort():
    def __init__(self):
        self.model = SiameseNetwork().to(DEVICE)
        self.model.load_state_dict(torch.load(sort_model_save_path))

        self.model.eval()
        self.qa_dict = json.load(open(QACORPUSPATH, 'r', encoding='utf-8'))

    def predict(self, sentence, recall_list):
        """

        :param sentence:{"cut","cut_by_word"}
        :param recall_list:
        :return:
        """
        # ws=None
        input1 = [cut(sentence['cut_by_word'], by_word=1, use_stopwords=1, with_sg=0)] * len(recall_list)
        input2 = [self.qa_dict[i]['q_cut_by_word'] for i in recall_list]
        input_output2_answer = {str(self.qa_dict[i]['q_cut_by_word']): self.qa_dict[i]['answer'] for i in recall_list}
        _input1 = torch.LongTensor([ws.transform(i, max_len=max_len_flag) for i in input1]).to(DEVICE)
        _input2 = torch.LongTensor([ws.transform(i, max_len=max_len_flag) for i in input2]).to(DEVICE)
        output = F.softmax(self.model(_input1, _input2), dim=-1)  # [batch_size,2]
        # print(output)
        # return
        output = output[:, -1].squeeze(-1).detach().cpu().numpy()
        print(list(zip(input2, output)))
        best_q, best_prob = sorted(list(zip(input2, output)), key=lambda x: x[1], reverse=True)[0]

        if best_prob > 0.75:
            print(best_q)
            return input_output2_answer[str(best_q)]
        else:
            return 'no res'


def create_ws():
    import pickle
    from dnn.sort.word_sequence import WordSequence
    ws = WordSequence()
    sort_q_content = open(q_cuted_words, 'r', encoding='utf-8').readlines()
    sort_sim_q_content = open(sim_q_cut_words, 'r', encoding='utf-8').readlines()
    for q_line in tqdm(sort_q_content):
        ws.fit(q_line.strip().split())
    for sim_q_line in tqdm(sort_sim_q_content):
        ws.fit(sim_q_line.strip().split())

    ws.build_vocab(min_count=5)
    print(len(ws))
    pickle.dump(ws, open(sort_ws, 'wb'))


if __name__ == '__main__':
    # train_run()
    # create_ws()
    sentence = {
        'cut_by_word': 'java 是 什 么',
        'cut': "java 是 什么",
        'entity': ['java']
    }
    resp = recall.predict(sentence)
    print(resp)
    dnn_sort = DnnSort()
    resp = dnn_sort.predict(sentence, resp)
    print(resp)
    pass
