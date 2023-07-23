#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 模型的训练
from tqdm import tqdm
from dnn.sort.dataset import data_loader
from dnn.sort.siamese import SiameseNetwork
from torch.optim import Adam
import torch.nn.functional as F
import torch
import numpy as np
from settings import DEVICE,sort_model_save_path,sort_optimizer_save_path

model = SiameseNetwork().to(DEVICE)
optimizer = Adam(model.parameters(), lr=0.001)


def train_run():
    for i in range(100):
        train(i)


def train(epoch):
    bar = tqdm(enumerate(data_loader), total=len(data_loader))
    loss_list = []
    for idx, (input1, input2, target) in bar:
        input1 = input1.to(DEVICE)
        input2 = input2.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        output = model(input1, input2)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        bar.set_description(f"{epoch}\t{idx}\t{np.mean(loss_list)}")
        if idx % 100 == 0:
            torch.save(model.state_dict(), sort_model_save_path)
            torch.save(optimizer.state_dict(), sort_optimizer_save_path)
