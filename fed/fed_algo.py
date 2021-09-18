import copy
from scipy import linalg
import numpy as np
import torch, random
import torch.nn.functional as F
from torch import nn


def fedavg(w_clients, dp=0):
    """
    Federated averaging
    :param w: list of client model parameters
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_avg = copy.deepcopy(w_clients[0])
    for k in w_avg.keys():
        for i in range(1, len(w_clients)):
            w_avg[k] = w_avg[k] + w_clients[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_clients)) + torch.mul(torch.randn(w_avg[k].shape), dp)
    return w_avg


def aggregate_att(w_clients, w_server, stepsize,  metric=2, dp=0):
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param stepsize: step size for aggregation
    :param metric: default setting is Frobenius norm. https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_next = copy.deepcopy(w_server)
    att, att_mat = {}, {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.norm(w_server[k]-w_clients[i][k], p=metric)
    for k in w_next.keys():
        att[k] = F.softmax(att[k], dim=0)
    for k in w_next.keys():
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
            w_next[k] = w_server[k] - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(w_server[k].shape), dp)
    return w_next


