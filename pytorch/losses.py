import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    y_pred = output_dict['clipwise_output']
    y_true = target_dict['target']
    if type(y_pred) is np.ndarray:
        y_pred = torch.from_numpy(y_pred)
    if type(y_true) is np.ndarray:
        y_true = torch.from_numpy(y_true)
    # print('y_true', type(y_true), y_true)

    # return nn.BCEWithLogitsLoss()(output_dict['clipwise_output'], target_dict['target'])
    return F.binary_cross_entropy(y_pred, y_true)


def clip_ce(output_dict, target_dict):
    """Cat. crossentropy loss.
    """
    y_pred = output_dict['clipwise_output']
    y_true = target_dict['target']

    if type(y_pred) is np.ndarray:
        y_pred = torch.from_numpy(y_pred)
    if type(y_true) is np.ndarray:
        y_true = torch.from_numpy(y_true)

    y_true = torch.squeeze(y_true)
    if y_true.dim() > 1:
        y_true = torch.argmax(y_true, dim=-1)

    # print('clip_ce y_true', y_true)
    # print('clip_ce y_pred', y_pred)

    return nn.CrossEntropyLoss()(y_pred, y_true)
    # F.cross_entropy(
    #     output_dict['clipwise_output'], target_dict['target'])


def clip_nll(output_dict, target_dict):
    y_pred = output_dict['clipwise_output']
    y_true = target_dict['target']

    if type(y_pred) is np.ndarray:
        y_pred = torch.from_numpy(y_pred)
    if type(y_true) is np.ndarray:
        y_true = torch.from_numpy(y_true)

    loss = - torch.mean(y_true * y_pred)
    return loss


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'clip_ce':
        return clip_ce
    elif loss_type == 'clip_nll':
        return clip_nll

