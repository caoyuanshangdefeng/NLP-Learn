#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch
from torch.utils.data import DataLoader, Dataset
from settings import sim_q_cut_words,q_cuted_words,sort_similarity,sort_ws
import pickle

max_len_flag = 20
ws = pickle.load(open(sort_ws, 'rb'))


class DnnSortDataset(Dataset):
    def __init__(self):
        self.q_lines = open(q_cuted_words, 'r', encoding='utf-8').readlines()
        self.sim_q_lines = open(sim_q_cut_words, 'r', encoding='utf-8').readlines()
        # 获取的数据认为前一半是相似的，后一半是不相似的
        self.target_lines = open(sort_similarity, 'r', encoding='utf-8').readlines()

        assert len(self.q_lines) == len(self.sim_q_lines) == len(self.target_lines), "dnn sort data length diff"

        pass

    def __getitem__(self, idx):
        q = self.q_lines[idx].split()
        sim_q = self.sim_q_lines[idx].split()
        target = int(self.target_lines[idx])
        # print(q, sim_q, len(q), len(sim_q))
        # len_q = len(q) if len(q) < max_len_flag else max_len_flag
        # len_sim_q = len(sim_q) if len(sim_q) < max_len_flag else max_len_flag

        # return q, sim_q,target, len_q, len_sim_q
        return q, sim_q, target

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
    input1, input2, target = zip(*batch)
    input1 = torch.LongTensor([ws.transform(i, max_len=max_len_flag) for i in input1])
    input2 = torch.LongTensor([ws.transform(i, max_len=max_len_flag) for i in input2])
    target = torch.LongTensor(target)
    return input1, input2, target


data_loader = DataLoader(dataset=DnnSortDataset(),
                         batch_size=128,
                         shuffle=True,
                         collate_fn=collate_fn
                         )
