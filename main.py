# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from model import GTN
from utils import get_metrics
from datalode import data_lode
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--kfold', default=10, type=int)
    parser.add_argument('--index', default=1, type=int)

    parser.add_argument('--num_channels', default=4, type=int)
    parser.add_argument('--num_layers', default=3, type=int)

    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    parser.add_argument('-t', default='o', type=str)

    args = parser.parse_args()
    print(args)

    drug_num = 708
    protein_num = 1512

    for i in range(10):
        args.index = i + 1
        A, DTI, protein_structure, drug_structure, train_label, test_label, one_index, zero_index, count = data_lode(
            args)

        model = GTN(num_edge=A.shape[0],
                    num_channels=args.num_channels,
                    num_layers=args.num_layers - 1,
                    drug_num=708,
                    protein_num=1512)

        class Myloss(nn.Module):
            def __init__(self):
                super(Myloss, self).__init__()

            def forward(self, iput, target):
                loss_sum = torch.pow((iput - target), 2)
                result = ((target * loss_sum).sum()) + (((1 - target) * loss_sum).sum())
                return (result)

        myloss = Myloss()
        layers_params = list(map(id, model.layers.parameters()))
        base_params = filter(lambda p: id(p) not in layers_params,
                             model.parameters())
        opt = torch.optim.Adam([{'params': base_params},
                                {'params': model.layers.parameters(), 'lr': 0.5}]
                               , lr=args.lr, weight_decay=args.weight_decay)

        print(f'The {i + 1} fold')
        for epoch in range(args.epoch):
            for param in opt.param_groups:
                if param['lr'] > 0.001:
                    param['lr'] *= 0.9

            model.train()
            opt.zero_grad()
            predict, Ws, att = model(A, DTI, drug_num, protein_num, protein_structure, drug_structure)

            loss = myloss(predict, train_label)
            print(f'epoch:  {epoch + 1}    train_loss:  {loss}')

            loss.backward()
            opt.step()

            with torch.no_grad():
                predict_test = predict.detach().cpu().numpy()
                predict_test_negative = predict_test[zero_index[0], zero_index[1]]
                predict_test_positive = predict_test[one_index[0], one_index[1]]

                predict_test_fold = np.concatenate((predict_test_positive, predict_test_negative))
                metrics = get_metrics(test_label, predict_test_fold)

                print('metrics:', metrics)
                print(f'AUPR: {metrics[0]:.4f}   AUC:  {metrics[1]:.4f}')


if __name__ == '__main__':
    main()