#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : eval.py
from tqdm import tqdm
from dnn.sort.dataset import data_loader as test_data_loader
from dnn.sort.siamese import SiameseNetwork
from torch.optim import Adam
import torch.nn.functional as F
import torch
import numpy as np
from settings import DEVICE
sort_model_save_path=r"E:\learn\AI\nlp_pro\nlp_learn\model\sort\model.pkl"
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load(sort_model_save_path))
optimizer = Adam(model.parameters(), lr=0.001)


def run():
    for i in range(1):
        eval(i)

model.eval()
def eval(epoch):
    bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader))
    loss_list = []
    acc_list=[]
    for idx, (input1, input2, target) in bar:
        input1 = input1.to(DEVICE)
        input2 = input2.to(DEVICE)
        target = target.to(DEVICE)
        output = model(input1, input2)
        loss = F.nll_loss(output, target)
        loss_list.append(loss.item())
        # 准确率
        pred=torch.max(output,dim=-1)[-1]
        acc=pred.eq(target).float().mean()
        acc_list.append(acc)
    print(np.mean(loss_list),np.mean(acc_list))




