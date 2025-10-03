import numpy as np
import torch
from torch.nn import functional as F

def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1)
    logits = logits.to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


def support_to_scalar(logits):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=-1)#进行 softmax 操作，将其转换为概率分布,dim=-1 表示在最后一个维度上进行 softmax 操作。
    support_size = logits.shape[-1]//2
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=-1)#终，函数返回了计算得到的标量值 x，这个值表示根据输入 logits 的概率分布计算得到的支持集的期望值。
    return x#这个函数的目的是将离散的分类表示（比如动作空间中的概率分布）转换为一个期望值输出比较具体的数值。


def logit_regression_loss(pred, true, mask=None):#计算对数几率回归损失（logit regression loss）的函数
    support_size = (pred.shape[-1] - 1) // 2
    if len(true.shape) == 3:
        b,n,l = true.shape
        true_dist = scalar_to_support(true.flatten(0,1), support_size)  # process positive and negative values
        true_dist = true_dist.reshape(*([b,n]+list(true_dist.shape[1:])))#升维
    else:#由于 softmax 输出的概率分布范围在 [0, 1] 之间，因此需要对真实标签进行转换，将其转换为支持集的形式。将正类别的真实标签映射到一个接近于 1 的值
        true_dist = scalar_to_support(true, support_size)  # process positive and negative values
    pred_logprob = F.log_softmax(pred, dim=-1)
    loss_all = -(true_dist * pred_logprob).mean(dim=-1)  # cross entropy (b, n, l)
    if mask is None:
        loss = loss_all.mean()
    else:
        loss = (loss_all * mask).sum() / mask.sum().clip(1)
    return loss


def logit_regression_mae(pred, true, mask=None):
    pred = support_to_scalar(pred)
    mae_all = torch.abs(true-pred)
    if mask is None:
        mae = mae_all.mean()
    else:
        mae = (mae_all*mask).sum()/mask.sum().clip(1)
    return mae
def cross_entropy(pred,action):
    n_len = pred.shape[-1]#最后一个向量大小
    true_action = action.expand_as(pred[..., 0]).clone()#pred_policy_train会有40个选择
    true_action_mask = true_action > 0#true
    #true_action_mask = true_action_mask * step_mask
    true_action[~true_action_mask] = -100#过大化无效标签 把0都变成-100


    loss_action = F.cross_entropy(pred.reshape(-1, n_len), true_action.reshape(-1),
                                    ignore_index=-100)#真实值不需要转标签
    return loss_action,true_action
def cross_entropy_mae(pred,true_action):
    # metrics action
    pred_action_scalar = pred.argmax(dim=-1)
    action_mae_all = torch.abs(pred_action_scalar - true_action)
    valid = true_action.expand_as(action_mae_all) > 0
    action_mae = (action_mae_all * valid).sum() / valid.sum().clip(1)
    return action_mae


def masked_loss(loss_func, pred, true, mask=None):
    mask = true >= 0
    if mask is not None:
        if pred.shape > true.shape:
            true = true.unsqueeze(-1) 
        else:
            pred = pred.expand_as(true)#计算损失时仅考虑掩码为 True 的位置上的样本。
        losses = loss_func(pred, true, reduction='none')
        loss = (losses * mask).sum() / mask.sum().clip(1)
    else:
        loss = loss_func(pred, true, reduction='mean')
    return loss


def masked_mean(values, mask=None):
    if mask is not None:
        mean = (values * mask).sum() / mask.sum().clip(1)
    else:
        mean = values.mean()
    return mean