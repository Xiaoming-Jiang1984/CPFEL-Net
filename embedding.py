import datasets_custom.dataset_alive as datasets
import models.modules as modules
from models.ABMIL import GatedAttention
import os
import sys
from tqdm import tqdm
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils_loss import CoxPHLoss
from lifelines.utils import concordance_index
from torch import Tensor

parser = argparse.ArgumentParser()

#I/O PARAMS
parser.add_argument('--output', type=str, default='/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/PAAD_10/result', help='name of output directory')
parser.add_argument('--data', type=str, default='/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/PAAD_10', help='path to the directory containing PT files')
parser.add_argument('--log', type=str, default='convergence_alive.csv', help='name of log file')
parser.add_argument('--encoder', type=str, default='exaonepath', choices=[
    'phikon',
    'uni',
    'exaonepath',
    'Hibou_b',
    'multi'
], help='which encoder to use')
parser.add_argument('--n_classes', type=int, default=1, help='number of classes (default: 2)')
parser.add_argument('--nidm', type=int, default=1536, help='number of features per model (default: 1536)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training (default: 8)')
parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='GPU ids used for training (default: 0)')
parser.add_argument('--is_fold', default=True, help='whether to use k-fold cross validation (default: True)')
#OPTIMIZATION PARAMS
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument("--lr", default=1e-4, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training).""")
parser.add_argument('--lr_end', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
parser.add_argument("--warmup_epochs", default=5, type=int, help="Number of epochs for the linear learning-rate warm up.")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
parser.add_argument('--weight_decay_end', type=float, default=0.1, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs (default: 40)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 10)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

def main(fold):
    # Get user input
    global args
    args = parser.parse_args()
    print("encoder:", args.encoder)

    # Dim of features
    if args.encoder == 'exaonepath':
        args.ndim = 768
        args.encoder = 'exaonepath'
    elif args.encoder == 'phikon':
        args.ndim = 768
    elif args.encoder == 'uni':
        args.ndim = 1024
        args.encoder = 'designeduni_freeze'  # For compatibility with the model
    elif args.encoder == 'virchow':
        args.ndim = 2560
    elif args.encoder == 'Prov_GigaPath':
        args.ndim = 1536
        args.encoder = 'Prov_GigaPath'  # For compatibility with the model
    elif args.encoder == 'H-optimus-0':
        args.ndim = 1536
        args.encoder = 'H_optimus_0'  # For compatibility with the model
    elif args.encoder == 'Hibou_b':
        args.ndim = 768
    elif args.encoder == 'multi':
        args.ndim = 1536

    def collate_fn_padded(batch):
        """
        使用填充使batch中所有样本的patch数量相同
        """
        inputs = []
        times = []
        events = []
        
        # 找到最大的patch数量
        max_patches = 0
        for item in batch:
            input_data, (time_val, event_val) = item
            feature = input_data
            max_patches = max(max_patches, feature.shape[0])
        
        padded_features = []

        for item in batch:
            input_data, (time_val, event_val) = item
            feature = input_data
            N = feature.shape[0]
            D = feature.shape[1]
            
            # 填充到最大长度
            if N < max_patches:
                pad_size = max_patches - N
                padding = torch.zeros(pad_size, D)
                
                feature_padded = torch.cat([feature, padding], dim=0)
            else:
                feature_padded = feature
                
            padded_features.append(feature_padded)
            times.append(time_val)
            events.append(event_val)
        
        # 堆叠成tensor
        feature_batch = torch.stack(padded_features, dim=0)  # [B, max_patches, D]
        
        return feature_batch, (torch.tensor(times), torch.tensor(events))

    # Set datasets
    train_dset, val_dset = datasets.get_datasets_single_kfold_one_size(args.data, encoder=args.encoder, seed=args.seed, fold=fold)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn_padded)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn_padded)

    # Get model
    gma = GatedAttention(in_dim=args.ndim, out_dim=args.n_classes)  

    if len(args.gpu) == 0:
        device = torch.device('cpu')
        gma = gma.to(device)

    elif len(args.gpu) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu[0])
        device = torch.device(f'cuda:{args.gpu[0]}')
        gma = gma.to(device)

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))
        device = torch.device('cuda')
        gma = nn.DataParallel(gma, device_ids=list(range(len(args.gpu))))
        # gma = nn.DataParallel(gma, device_ids=args.gpu)
        gma = gma.cuda()
    print('run on gpu:', device)
    
    # Set loss
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = CoxPHLoss(alpha=1e-6).to(device)  # 1e-6

    # Set optimizer
    params_groups = get_params_groups(gma)
    optimizer = optim.AdamW(params_groups)
    
    # Set schedulers
    lr_schedule = cosine_scheduler(
        args.lr,
        args.lr_end,
        args.nepochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.nepochs,
        len(train_loader),
    )
    cudnn.benchmark = True
        
    # Set output files
    if args.is_fold:
        file_save_pth = os.path.join(args.output, args.encoder, 'alive_one_size_kfold_20', str(fold))
    else:
        file_save_pth = os.path.join(args.output, args.encoder, 'alive_one_size_20', str(fold))
    if not os.path.exists(file_save_pth):
        os.makedirs(file_save_pth)
    log_save_pth = os.path.join(file_save_pth, args.log)
    with open(log_save_pth, 'w') as fconv:
        fconv.write('epoch,metric,value\n')

    # best_auc = 0
    best_cindex = 0
    best_epoch = 0
    # Main training loop
    for epoch in range(args.nepochs+1):
        
        ## Training logic
        # if epoch > 0:
        loss = train(epoch, train_loader, gma, criterion, optimizer, lr_schedule, wd_schedule, args, device)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch, args.nepochs, loss))
        with open(log_save_pth, 'a') as fconv:
            fconv.write('{},loss,{}\n'.format(epoch, loss))
        clean_memory()
        
        ## Validation logic
        risk_scores, times, events = test(args, epoch, val_loader, gma, device)

        for name, param in gma.named_parameters():
            if torch.isnan(param).any():
                print("⚠️ NaN detected in parameter:", name)

        if isinstance(risk_scores, np.ndarray):
            if np.isnan(risk_scores).any():
                print("⚠️ NaN detected in risk_scores (numpy)")
        else:
            if torch.isnan(risk_scores).any():
                print("⚠️ NaN detected in risk_scores (tensor)")

        cindex = concordance_index(times, risk_scores, events)  
        clean_memory()
    
        ### Printing stats
        print('Validation\tEpoch: [{}/{}]\tC-index: {}'.format(epoch, args.nepochs, cindex))
        with open(log_save_pth, 'a') as fconv:
            fconv.write('{},cindex,{}\n'.format(epoch, cindex))
        
        ### Model saving logic
        obj = {
            'epoch': epoch,
            'state_dict': gma.state_dict(),
            'cindex': cindex,
            'optimizer': optimizer.state_dict()
        }
        torch.save(obj, os.path.join(file_save_pth, 'checkpoint.pth'))

        if cindex > best_cindex:
            best_cindex = cindex
            best_epoch = epoch
            torch.save(obj, os.path.join(file_save_pth, 'best.pth'))
            print(f"Best model saved at epoch {epoch} with C-index: {cindex}")
    
    # 日志末尾追加最终最佳指标
    with open(log_save_pth, 'a') as f:
        f.write(f'best: epoch: {best_epoch}, C-index: {best_cindex:.4f}\n')
    print(f'\n=== Fold {fold} Final Best Metrics ===')
    print(f'Best Epoch: {best_epoch}, Best C-index: {best_cindex:.4f}')

def test(args, run, loader, model, device):
    # Set model in test mode
    model.eval()

    risk_scores_all = []
    times_all = []
    events_all = []

    with torch.no_grad():
        for i, (input, (time, event)) in enumerate(tqdm(loader, desc=f'Inference Epoch {run}/{args.nepochs}', ncols=100)):
            ## Forward pass
            _, output = model(input, device= device)
    
            risk_scores_all.append(output.cpu())
            times_all.append(time)
            events_all.append(event)
            torch.cuda.empty_cache()
            
    # 合并并转换为numpy
    risk_scores_all = torch.cat(risk_scores_all).numpy()
    times_all = torch.cat(times_all).numpy()
    events_all = torch.cat(events_all).numpy()
    
    return risk_scores_all, times_all, events_all


def train(run, loader, model, criterion, optimizer, lr_schedule, wd_schedule, args, device):
    model.train()
    running_loss = 0.
    pbar = tqdm(total=len(loader), desc=f'Training Epoch {run}', ncols=100)

    for i, (input, (time, event)) in enumerate(loader):
        # optimizer.zero_grad()
        # 更新学习率
        it = len(loader) * (run-1) + i
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if j == 0:
                param_group["weight_decay"] = wd_schedule[it]

        time = time.float().to(device) 
        event = event.float().to(device) 
        
        # 前向传播 - 使用no_grad减少中间变量的保留
        # with torch.cuda.amp.autocast():  # 混合精度训练
        _, output = model(input, device=device)

        loss = criterion(output, event, time)
        
        # 反向传播
        loss.backward()
        
        # 优化步骤
        optimizer.step()
        optimizer.zero_grad()
    
        # 记录损失
        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)

        # 更新进度条
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'AvgLoss': f"{avg_loss:.4f}",
        })

        # 及时释放
        del output, loss
        torch.cuda.empty_cache()

        pbar.update(1)
    
    pbar.close()
    print(f"Epoch {run} finished | Average Loss = {avg_loss:.4f}")
    return avg_loss

def clean_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


if __name__ == '__main__':
    for fold in range(5):
        print('fold:', fold)
        main(fold)
        print('------------------------')