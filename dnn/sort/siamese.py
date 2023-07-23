#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : siamese.py

"""
构建孪生神经网络
process:
1.embedding
2.GRU
3.attention
4.attention concat gru output
5.GRU
6.pooling
7.DNN
"""
import pickle

sort_hidden_size = 256
sort_num_layers = 2
sort_drop_out = 0.3
bidirectional = True
import torch.nn as nn
import torch.nn.functional as F
import torch
# from dnn.sort.word_sequence import WordSequence
from settings import sort_ws
ws = pickle.load(open(sort_ws, 'rb'))
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(ws),
                                      embedding_dim=300,
                                      padding_idx=ws.PAD)
        self.gru1 = nn.GRU(
            input_size=300,
            hidden_size=sort_hidden_size,
            num_layers=sort_num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.gru2 = nn.GRU(
            input_size=sort_hidden_size * 4,
            hidden_size=sort_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        #
        self.dnn = nn.Sequential(
            nn.Linear(sort_hidden_size * 4, sort_hidden_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(sort_hidden_size),
            nn.Dropout(sort_drop_out),

            nn.Linear(sort_hidden_size, sort_hidden_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(sort_hidden_size),
            nn.Dropout(sort_drop_out),

            nn.Linear(sort_hidden_size, 2),  # 2 分类模型，相似不相似
            # nn.Softmax(dim=-1)
        )

        # 18:36

    def forward(self, input1, input2):
        mask1, mask2 = input1.eq(ws.PAD), input2.eq(ws.PAD)  # [batch_size,max_len]
        # embedding
        input1 = self.embedding(input1)  # [batch_size,max_len,300]
        input2 = self.embedding(input2)
        # GRU
        # output1 :[batch_size,max_len,hidden_size*num_layer]
        # hidden_state1:[num_layer*2,batch_size,hidden_size]
        output1, hidden_state1 = self.gru1(input1)
        output2, hidden_state2 = self.gru1(input2)
        output1_align, output2_align = self.soft_attention_align(output1, output2, mask1, mask2)
        output1 = torch.cat([output1, output1_align], dim=-1)  # [batch_size,max_len,hidden_size*num_layer*2]
        output2 = torch.cat([output2, output2_align], dim=-1)

        # GRU

        gru2_output1, _ = self.gru2(output1)  # [batch_size,max_len,batch_size*1]
        gru2_output2, _ = self.gru2(output2)
        # pooling
        output1_pooled = self.apply_pooling(gru2_output1)  # [batch_size,batch_size,2]
        output2_pooled = self.apply_pooling(gru2_output2)

        out = torch.cat([output1_pooled, output2_pooled], dim=-1)  # [batch_size,hidden_size*4]
        out = self.dnn(out)  # [batch_size,2]
        return F.log_softmax(out, dim=-1)

    def apply_pooling(self, output):
        # 最后一个维度除以步长
        avg_pooled = F.avg_pool1d(output.transpose(1, 2), kernel_size=output.size(1)).squeeze(
            -1)  # [batch_size,hidden_size]
        max_pooled = F.max_pool1d(output.transpose(1, 2), kernel_size=output.size(1)).squeeze(-1)
        return torch.cat([avg_pooled, max_pooled], dim=-1)  # [batch_size,batch_size*2]

    def soft_attention_align(self, x1, x2, mask1, mask2):
        """self attention"""
        mask1 = mask1.float().masked_fill_(mask1, float("-inf"))
        mask2 = mask2.float().masked_fill_(mask2, float("-inf"))
        attention_weight = x1.bmm(x2.transpose(1, 2))
        x1_weight = F.softmax(attention_weight + mask2.unsqueeze(1), dim=-1)
        x2_output = x1_weight.bmm(x2)

        x2_weight = F.softmax(attention_weight.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x1_output = x2_weight.bmm(x1)

        """
        #  1.attention weight
        # x1 [batch_size,max_len,hidden_size*num_layer]
        # x2 [batch_size,hidden_size*num_layer,max_len]
        # x2 encode
        x1_weight = x1.bmm(x2.transpose(1, 2) + mask2.unsqueeze(1))  # [batch_size,x1_len,x2_len]

        #  2.attention weight * output
        x1_weight = F.softmax(x1_weight, dim=-1)
        x2_output = x1_weight.bmm(x2)  # [batch_size,seq_len,hidden_size*num_layer]

        x2_weight = x1_weight.transpose(1, 2)
        x1_output = x2_weight.bmm(x1)
        """

        return x1_output, x2_output


if __name__ == '__main__':
    conn = SiameseNetwork()
