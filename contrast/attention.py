import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import datasets_custom.dataset_classify as datasets
from models.ours_classify import AdaptiveGatedAttention_ModelAttention
from tqdm import tqdm
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

parser = argparse.ArgumentParser()

# I/O PARAMS
parser.add_argument('--output', type=str, default='/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/contrast/result/Attention/Camelyon16/Camelyon16_20/result', help='name of output directory')
parser.add_argument('--data', type=str, default='/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/Camelyon16/Camelyon16_20', help='path to the directory containing PT files')
parser.add_argument('--log', type=str, default='convergence_classify.csv', help='name of log file')
parser.add_argument('--encoder', type=str, default='multi', choices=['phikon', 'uni', 'exaonepath', 'Hibou_b', 'multi'], help='which encoder to use')
parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--gpu', type=int, default=[1], nargs='+', help='GPU ids used for training')
parser.add_argument('--is_fold', default=True, help='whether to use k-fold cross validation')

# OPTIMIZATION PARAMS
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate at the end of linear warmup")
parser.add_argument('--lr_end', type=float, default=1e-6, help="Target LR at the end of optimization")
parser.add_argument("--warmup_epochs", default=5, type=int, help="Number of epochs for linear learning-rate warm up")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="Initial value of the weight decay")
parser.add_argument('--weight_decay_end', type=float, default=0.1, help="Final value of the weight decay")
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=42, help='random seed')

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
    
    def collate_fn_padded(batch, device='cpu'):
        B = len(batch)

        # 读取第一个样本的结构
        first_models = batch[0][0]       # [num_models, N, D]
        num_models = first_models.shape[0]
        D = first_models.shape[2]

        # 计算 batch 内最大 patch 数
        max_patch = max(models.shape[1] for (models, _) in batch)

        # 分配批量 tensor
        features_batch = torch.zeros(
            (B, num_models, max_patch, D),
            dtype=torch.float32,
            device=device
        )

        labels = []

        # 填充所有模型特征
        for b_idx, (models, label) in enumerate(batch):
            N = models.shape[1]              # 当前样本 patch 数
            features_batch[b_idx, :, :N, :] = models

            labels.append(label)

        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        return features_batch, labels


    # Load datasets
    train_dset, val_dset = datasets.get_datasets_multi_one_size_kfold(args.data, encoder=args.encoder, seed=args.seed, fold=fold)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn_padded)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn_padded)

    # Initialize model
    model = AdaptiveGatedAttention_ModelAttention(in_dim=args.ndim, out_dim=args.n_classes)  

    # Set device
    if args.gpu:
        if len(args.gpu) == 1:
            device = torch.device(f'cuda:{args.gpu[0]}')
            model = model.to(device)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))
            device = torch.device('cuda:0')
            model = model.to(device)
            model = nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    print('run on gpu:', device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    params_groups = get_params_groups(model)
    optimizer = optim.AdamW(params_groups)
    
    # Schedulers
    lr_schedule = cosine_scheduler(args.lr, args.lr_end, args.nepochs, len(train_loader), warmup_epochs=args.warmup_epochs)
    wd_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end, args.nepochs, len(train_loader))
    cudnn.benchmark = True
        
    # Create output directory
    save_dir = os.path.join(args.output, args.encoder, 'classify_one_size_kfold_20' if args.is_fold else 'classify_one_size_20', str(fold))
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize log file
    log_path = os.path.join(save_dir, args.log)
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,accuracy,f1,auc\n')

    # Training setup
    best_auc = 0.0  
    best_acc = 0.0
    best_epoch = 0
    
    # Main training loop
    for epoch in range(args.nepochs + 1):
        train_loss = 0.0
        # # 训练逻辑：epoch>0才训练，但epoch0也要执行验证和保存判断
        # if epoch > 0:
        train_loss = train_epoch(epoch, train_loader, model, criterion, optimizer, lr_schedule, wd_schedule, args, device)
        print(f'Training\tEpoch: [{epoch}/{args.nepochs}]\tLoss: {train_loss:.4f}')
        clean_memory()
    
        # 验证逻辑：所有epoch都执行（包括epoch0）
        val_metrics = validate(epoch, val_loader, model, criterion, device, args.n_classes)
        
        # 验证指标输出
        val_log_str = (
            f'Validation\tEpoch: [{epoch}/{args.nepochs}]\tval_loss: {val_metrics["val_loss"]:.4f} | '
            f'accuracy: {val_metrics["accuracy"]:.4f} | f1: {val_metrics["f1"]:.4f} | auc: {val_metrics["auc"]:.4f}'
        )
        print(val_log_str)
        
        current_best_updated = False
        # 放宽判断条件：AUC更高 或 AUC相同但Acc更高 或 初始状态（best_auc=-1）
        if (val_metrics['auc'] > best_auc) or \
           (abs(val_metrics['auc'] - best_auc) < 1e-6 and val_metrics['accuracy'] > best_acc):  
            best_auc = val_metrics['auc']
            best_acc = val_metrics['accuracy']
            best_epoch = epoch
            current_best_updated = True

            print(f"Best model saved at epoch {epoch} | AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # 日志记录
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{train_loss:.4f},{val_metrics["val_loss"]:.4f},{val_metrics["accuracy"]:.4f},{val_metrics["f1"]:.4f},{val_metrics["auc"]:.4f}\n')
        
        # 保存模型：所有epoch都执行（包括epoch0）
        save_model(model, optimizer, epoch, val_metrics, save_dir, 
                   val_metrics['auc'], val_metrics['accuracy'], 
                   best_auc, best_acc, best_epoch, current_best_updated)

    # 日志末尾追加最终最佳指标
    with open(log_path, 'a') as f:
        f.write(f'best,{best_epoch}, AUC: {best_auc:.4f}, Acc: {best_acc:.4f}\n')
    print(f'\n=== Fold {fold} Final Best Metrics ===')
    print(f'Best Epoch: {best_epoch} | Best AUC: {best_auc:.4f} | Best Acc: {best_acc:.4f}')

def train_epoch(epoch, loader, model, criterion, optimizer, lr_schedule, wd_schedule, args, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(total=len(loader), desc=f'Training Epoch {epoch}', ncols=100)
    for i, (inputs, labels) in enumerate(loader):
        it = len(loader) * (epoch - 1) + i
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if j == 0:
                param_group["weight_decay"] = wd_schedule[it]
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        if labels.dim() > 1:
            labels = labels.squeeze()
        optimizer.zero_grad()
        _, outputs = model(inputs, device=device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'AvgLoss': f'{avg_loss:.4f}'}, refresh=True)
        pbar.update(1)
        del outputs, loss
        torch.cuda.empty_cache()
    pbar.close()
    return avg_loss

def validate(epoch, loader, model, criterion, device, n_classes):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    val_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Validation Epoch {epoch}', ncols=100)
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            if labels.dim() > 1:
                labels = labels.squeeze()
            _, outputs = model(inputs, device=device)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1] if n_classes == 2 else probs)
            all_labels.extend(labels.cpu().numpy())
            torch.cuda.empty_cache()
        pbar.close()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    if all_probs.ndim > 1 and n_classes == 2:
        all_probs = all_probs[:, 1]
    metrics = {
        'val_loss': val_loss / len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='binary' if n_classes == 2 else 'weighted', zero_division=0)
    }
    try:
        if n_classes == 2:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        else:
            metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        metrics['auc'] = 0.0
    return metrics

def save_model(model, optimizer, epoch, metrics, save_dir, current_auc, current_acc, best_auc, best_acc, best_epoch, current_best_updated):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'metrics': metrics,
        'optimizer': optimizer.state_dict(),
        'best_auc': best_auc,
        'best_acc': best_acc,
        'best_epoch': best_epoch
    }
    # 保存最新模型（所有epoch都保存）
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    # 保存最佳模型（仅当更新最佳时）
    if current_best_updated:
        torch.save(checkpoint, os.path.join(save_dir, 'best.pth'))
        # 这里的打印是兜底，主循环已强制打印epoch0的保存提示

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
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters) if warmup_epochs > 0 else np.array([])
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return np.concatenate((warmup_schedule, schedule))

if __name__ == '__main__':
    for fold in range(0, 5, 1):
        print('fold:', fold)
        main(fold)
        print('------------------------')