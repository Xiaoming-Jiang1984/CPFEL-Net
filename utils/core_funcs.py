import copy
import time
from argparse import Namespace
from collections import OrderedDict
import os
import pickle
import torch
from utils.utils import *
from utils.utils_loss import *
from torch.cuda.amp import GradScaler, autocast
import gc
import tqdm
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
import cv2
from utils.utils_metrics import *
from utils.utils_earlystop import *
import torch.nn.functional as F

def forward_model(model, datas, args, loss_fn, use_fp16=True):
    x, label = datas
    # print(x.shape, label.shape)
    # print(type(x), type(label))
    x = x.cuda()
    # label = torch.tensor(label).cuda()
    label = label.cuda()
    # print(x.shape, label.shape)


    # st = time.time()
    # print(x.shape, coords.shape)
    with torch.cuda.amp.autocast(enabled=use_fp16):
        pred = model(x)
    # ed = time.time()
    # print('model forward cost', ed - st, 's')
    # print(pred.shape)

    if loss_fn is not None:
        loss = loss_fn(pred, label)
    else:
        loss = None
    # print(loss)
    # print()

    return loss, pred, (x, label)


def train_loop_survival(epoch, model, loader, optimizer, args, writer=None,
                        loss_fn=None, gc=16, loss_scaler=None, bar=True):
    model.train()
    train_loss = 0.
    all_pred_scores, all_censorships, all_ground_truth = [], [], []
    if bar == True:
        loader = tqdm.tqdm(loader)

    for batch_idx, datas in enumerate(loader):
        if not datas:
            continue
        loss, pred, dts = forward_model(model, datas, args, loss_fn, use_fp16=args.use_fp16)
        if torch.any(torch.isnan(pred)):
            print('error: train out nan')
            model.zero_grad()
            # optimizer.zero_grad()
            continue
        if torch.isnan(loss):
            print('loss nan')
            model.zero_grad()
            # optimizer.zero_grad()
            continue
        x, label = dts

        loss_scaler(loss, optimizer, parameters=model.parameters())
        optimizer.zero_grad()

        loss_value = loss.item()
        if pred.shape[1] == 1:
            pred = pred.detach().cpu().numpy()
        else:
            pred = pred.detach()
            # pred = F.softmax(pred, dim=1)
            pred = pred.cpu().numpy()
        train_loss += loss_value

        # print('pred.shape', pred.shape)

        # all_pred_scores.extend(pred.flatten())
        all_pred_scores.extend(pred)
        all_ground_truth.extend(label.cpu().numpy().flatten())

    # calculate loss and error for epoch
    train_loss /= len(all_pred_scores)

    train_results = acc_metric_train(all_pred_scores, all_censorships, all_ground_truth,
                                     epoch, train_loss, num_class=args.num_classes)
    if writer:
        writer.add_scalar('train/acc', train_results['acc'], epoch)
        writer.add_scalar('train/bacc', train_results['bacc'], epoch)
        writer.add_scalar('train/aucroc', train_results['aucroc'], epoch)


def validate_survival(epoch, model, loader, args, early_stopping=None,
                      writer=None, results_dir=None, cur=None,
                      bar=True, mode='val'):
    _, metrics_dict = summary_survival(model, loader, args, bar=bar, mode=mode,
                                       get_patient_results=False, epoch=epoch, verbose=True)

    # in_met = metrics_dict['val_bacc']
    in_met = metrics_dict['val_loss']
    if writer:
        writer.add_scalar('val/ce_loss', metrics_dict['val_loss'], epoch)
        writer.add_scalar('val/acc', metrics_dict['val_acc'], epoch)
        writer.add_scalar('val/precision', metrics_dict['val_precision'], epoch)
        writer.add_scalar('val/aucroc', metrics_dict['val_aucroc'], epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, in_met, model,
                       ckpt_name=os.path.join(results_dir, f"best_checkpoint.pt"))

        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False


def summary_survival(model, loader, args, bar=True, mode='val', get_patient_results=True,
                     epoch='summary_survival', verbose=False):
    model.eval()
    all_pred_scores, all_censorships, all_ground_truth = [], [], []
    if bar == True:
        loader = tqdm.tqdm(loader)
    for batch_idx, datas in enumerate(loader):
        if not datas:
            continue
        with torch.no_grad():
            _, pred, dts = forward_model(model, datas, args, loss_fn=None, use_fp16=args.use_fp16)
        if torch.any(torch.isnan(pred)):
            print('error: val out nan')
            continue
        x, label = dts

        # pred = pred.cpu().numpy()
        if pred.shape[1] == 1:
            pred = pred.detach().cpu().numpy()
        else:
            pred = pred.detach()
            # pred = F.softmax(pred, dim=1)
            pred = pred.cpu().numpy()

        # all_pred_scores.extend(pred.flatten())
        all_pred_scores.extend(pred)
        all_ground_truth.extend(label.cpu().numpy())

    metrics_dict = acc_metric_val(all_pred_scores, all_censorships, all_ground_truth, epoch,
                                  num_class=args.num_classes, verbose=verbose, mode=mode)
    metrics_dict = OrderedDict(metrics_dict)
    return {}, metrics_dict


def summary_survival_test(model, loader, args, bar=True, mode='val', get_patient_results=True,
                     epoch='summary_survival', verbose=False):
    model.eval()
    all_pred_scores, all_censorships, all_ground_truth = [], [], []
    all_pred_results = []
    rt_disk = {'label': [], 'pred': []}
    if bar == True:
        loader = tqdm.tqdm(loader)
    for batch_idx, datas in enumerate(loader):
        if not datas:
            continue
        with torch.no_grad():
            _, pred, dts = forward_model(model, datas, args, loss_fn=None, use_fp16=args.use_fp16)
        if torch.any(torch.isnan(pred)):
            print('error: val out nan')
            continue
        x, label = dts

        # pred = pred.cpu().numpy()
        if pred.shape[1] == 1:
            pred = pred.detach().cpu().numpy()
        else:
            pred = pred.detach()
            # pred = F.softmax(pred, dim=1)
            pred = pred.cpu().numpy()

        # all_pred_scores.extend(pred.flatten())
        all_pred_scores.extend(pred)
        all_ground_truth.extend(label.cpu().numpy())

        
        # 根据预测的二分类结果的概率值，判断是否为阳性
        for i in range(len(all_pred_scores)):
            if all_pred_scores[i][1] > all_pred_scores[i][0]:
                all_pred_results.append(1)
            else:
                all_pred_results.append(0)
        
        # 保存预测结果和真实结果
        for i in range(len(all_ground_truth)):
            rt_disk['label'].append(all_ground_truth[i])
            rt_disk['pred'].append(all_pred_results[i])

    metrics_dict = acc_metric_val(all_pred_scores, all_censorships, all_ground_truth, epoch,
                                  num_class=args.num_classes, verbose=verbose, mode=mode)
    metrics_dict = OrderedDict(metrics_dict)
    return rt_disk, metrics_dict


# def summary_survival_test(model, loader, args, bar=True, mode='val', get_patient_results=True,
#                      epoch='summary_survival', verbose=False):
#     model.eval()
#     all_pred_scores, all_censorships, all_ground_truth = [], [], []
#     all_pred_results = []
#     rt_disk = {'label': [], 'pred': []}
#     if bar == True:
#         loader = tqdm.tqdm(loader)
#     for batch_idx, datas in enumerate(loader):
#         if not datas:
#             continue
#         with torch.no_grad():
#             _, pred, dts = forward_model(model, datas, args, loss_fn=None, use_fp16=args.use_fp16)
#         if torch.any(torch.isnan(pred)):
#             print('error: val out nan')
#             continue
#         x, label = dts

#         # pred = pred.cpu().numpy()
#         if pred.shape[1] == 1:
#             pred = pred.detach().cpu().numpy()
#         else:
#             pred = pred.detach()
#             # pred = F.softmax(pred, dim=1)
#             pred = pred.cpu().numpy()

#         # all_pred_scores.extend(pred.flatten())
#         all_pred_scores.extend(pred)
#         all_ground_truth.extend(label.cpu().numpy())

        
#         # 根据预测的二分类结果的概率值，判断是否为阳性
#         for i in range(len(all_pred_scores)):
#             if all_pred_scores[i][1] > all_pred_scores[i][0]:
#                 all_pred_results.append(1)
#             else:
#                 all_pred_results.append(0)
        
#         # 保存预测结果和真实结果
#         for i in range(len(all_ground_truth)):
#             rt_disk['label'].append(all_ground_truth[i])
#             rt_disk['pred'].append(all_pred_results[i])

#     metrics_dict = acc_metric_val(all_pred_scores, all_censorships, all_ground_truth, epoch,
#                                   num_class=args.num_classes, verbose=verbose, mode=mode)
#     metrics_dict = OrderedDict(metrics_dict)
#     return rt_disk, metrics_dict

