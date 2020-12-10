import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd


def clip_nll(output_dict, target_dict):
    output = - torch.mean(target_dict['target'] * output_dict['clipwise_output'])

    # meta_train_df = pd.read_csv('/home/den/datasets/experiments/emocon_mult/meta_train.csv', sep=';')
    # loss = nn.CrossEntropyLoss(weight=torch.tensor(meta_train_df.arvalmix.value_counts().sort_index().values))
    # output = loss(torch.LongTensor(output_dict['clipwise_output']).to('cpu'), torch.LongTensor(target_dict['target']).to('cpu'))
    return output


def get_loss_func(loss_type):
    if loss_type == 'clip_nll':
        return clip_nll