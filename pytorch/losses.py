import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd


def clip_nll(output_dict, target_dict):
    loss = target_dict['target'] * output_dict['clipwise_output']
    loss = - torch.mean(loss)
    # loss = - torch.mean(loss*weight)
    return loss


def get_loss_func(loss_type):
    if loss_type == 'clip_nll':
        return clip_nll