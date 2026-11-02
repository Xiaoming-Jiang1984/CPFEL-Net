import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler

import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

from torch.utils.data.dataloader import default_collate
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# c 0确认死亡 1存活
def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]

    # uncensored_loss 确认死亡事件发生前面不为0+
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(
        torch.gather(hazards, 1, Y).clamp(min=eps)))

    # censored_loss 存活时不为0
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))

    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss  # alpha一般是0
    loss = loss.mean()
    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
        1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None, **kwargs):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None, **kwargs):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)
    # h_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    # reg = - (1 - c) * (torch.log(torch.gather(hazards, 1, Y)) + torch.gather(torch.cumsum(torch.log(1-h_padded), dim=1), 1, Y))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        print('cox loss alpha:', self.alpha)

    # event_time:T c:E
    def forward(self, S: Tensor, c: Tensor, event_time: Tensor, **kwargs) -> Tensor:
        return self.cox_ph_loss(S, event_time, c)

    def cox_ph_loss(self, log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
        """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.

        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
        idx = durations.sort(descending=True)[1]

        events = events[idx]
        log_h = log_h[idx]

        return self.cox_ph_loss_sorted(log_h, events, eps)

    def cox_ph_loss_sorted(self, log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
        """Requires the input to be sorted by descending duration time.
        See DatasetDurationSorted.

        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.

        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """

        if events.dtype is not torch.float32:
            events = events.float()

        events = 1 - events  ### censorship的值和event是反过来的
        events[events == 0] = self.alpha

        events = events.view(-1)
        log_h = log_h.view(-1)
        gamma = log_h.max()
        log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)

        return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


def l1_reg_all(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg


# subtype_dict = {
#     'LGG': ['Astrocytoma, anaplastic', 'Oligodendroglioma, anaplastic', 'Mixed glioma'],  # 268:359:216
#     'BRCA': ['Lobular carcinoma', 'Infiltrating duct carcinoma'],  # 204:805
#     'COAD': ['Mucinous adenocarcinoma', 'Adenocarcinoma']  # 63:389
# }
focal_alpha = {
    'BRCA': (0.8, 0.15),
    'LGG': (0.373, 0.278, 0.463),
    'COAD': (0.8, 0.2)
}
def get_loss(args):
    print('\nInit loss function...')
    print(args.task, args.bag_loss)
    if args.task == 'survival':
        if args.bag_loss == 'ce':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox':
            loss_fn = CoxPHLoss(alpha=args.alpha_surv)
        else:
            raise NotImplementedError
    else:
        if args.bag_loss == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        elif args.bag_loss == 'focalloss':
            print(args.project_name, 'focal_alpha', focal_alpha[args.project_name])
            loss_fn = MultiClassFocalLossWithAlpha(alpha=focal_alpha[args.project_name])
        else:
            raise NotImplementedError
    print('Done')
    return loss_fn


def composite_loss(risk_scores, time, event, time_points, alpha=0.5):
    # Cox部分似然损失
    def cox_ph_loss(risk_scores, time, event):
        order = torch.argsort(time)
        risk_scores = risk_scores[order]
        event = event[order]

        exp_scores = torch.exp(risk_scores)
        log_cumsum_exp = torch.log(torch.cumsum(exp_scores, dim=0))
        uncensored_likelihood = risk_scores - log_cumsum_exp
        censored_likelihood = uncensored_likelihood * event

        loss = -torch.mean(censored_likelihood)
        return loss

    # 生成二分类标签
    def create_binary_labels(time, time_points):
        labels = [(time <= tp).float() for tp in time_points]
        return labels

    binary_labels = create_binary_labels(time, time_points)

    # 时间依赖的AUC损失
    # def time_dependent_auc_loss(risk_scores, time, event, binary_labels, time_points):
    #     auc_losses = []
    #     for tp, labels in zip(time_points, binary_labels):
    #         mask = time <= tp
    #         if torch.sum(mask) > 0:
    #             if len(torch.unique(labels[mask])) > 1:  # 检查是否有超过一个类别
    #                 auc = roc_auc_score(labels[mask].cpu().numpy(), risk_scores[mask].detach().cpu().numpy())
    #                 auc_loss = 1 - auc
    #                 auc_losses.append(auc_loss)
    #     return torch.tensor(auc_losses).mean() if auc_losses else torch.tensor(0.0)

    # 时间依赖的二分类损失
    def time_dependent_binary_loss(risk_scores, binary_labels, time_points):
        bce = nn.BCEWithLogitsLoss()
        binary_losses = []
        for tp, labels in zip(time_points, binary_labels):
            mask = (time <= tp) | (event == 1)  # 仅考虑截尾或事件发生的样本
            if torch.sum(mask) > 0:
                loss = bce(risk_scores[mask], labels[mask])
                binary_losses.append(loss)
        return torch.tensor(binary_losses).mean() if binary_losses else torch.tensor(0.0)

    # 综合损失
    cox_loss = cox_ph_loss(risk_scores, time, event)
    # auc_loss = time_dependent_auc_loss(ris


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=(0.2, 0.3, 0.5), gamma=2, reduction='mean', device='cuda'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = torch.tensor(gamma).cuda()
        self.reduction = reduction

    def forward(self, pred, target):

        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1)  # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
