#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/16 17:59
# @Author  : 草原上的风
# @File    : dataset.py

import os

import torch
from torch.utils.data import DataLoader, Dataset
from settings import SORTINPUTDATA, SORTRESP
import pickle
from dnn.sort.word_sequence import WordSequence

sort_ws_path = r"E:\learn\AI\nlp_pro\nlp_learn\model\sort\sort_ws.pkl"
ws = pickle.load(open(sort_ws_path, 'rb'))


class DnnSortDataset(Dataset):
    def __init__(self):
        self.q_lines = open(os.path.join(SORTINPUTDATA, "q_cuted_words.txt"), 'r', encoding='utf-8').readlines()
        self.sim_q_lines = open(os.path.join(SORTINPUTDATA, "sim_q_cuted_words.txt"), 'r', encoding='utf-8').readlines()
        assert len(self.q_lines) == len(self.sim_q_lines), "dnn sort data length diff"

        pass

    def __getitem__(self, idx):
        q = self.q_lines[idx].split()
        sim_q = self.sim_q_lines[idx].split()
        print(q, sim_q, len(q), len(sim_q))

        return q, sim_q, len(q), len(sim_q)
        pass

    def __len__(self):
        return len(self.q_lines)


def collate_fn(batch):
    """
    max_len 问题的最大长度

    :param batch:
    :return:
    """
    batch = sorted(batch, key=lambda x: x[-2], reverse=True)
    print(batch)
    input, target, input_length, target_length = zip(*batch)
    input = [ws.transform(i, max_len=20) for i in input]
    input = torch.LongTensor(input)
    target = [ws.transform(i, max_len=20) for i in target]
    target = torch.LongTensor(target)
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)
    return input, target, input_length, target_length


data_loader = DataLoader(dataset=DnnSortDataset(),
                         batch_size=128,
                         shuffle=True,
                         collate_fn=collate_fn
                         )
