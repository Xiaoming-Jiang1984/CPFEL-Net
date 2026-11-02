import os
import numpy as np
import pandas as pd
import torch


class slide_dataset_survival(object):
    '''
    Slide level dataset for survival analysis which returns:
    - feature matrices (h)
    - survival time
    - event status (1=event occurred, 0=censored)
    '''
    def __init__(self, df, encoder, pth):
        self.df = df
        self.pth = pth
        self.encoder = encoder
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        feature1 = torch.load(os.path.join(self.pth, row.pt + '_5.pt'))
        feature2 = torch.load(os.path.join(self.pth, row.pt + '_10.pt'))
        feature3 = torch.load(os.path.join(self.pth, row.pt + '_20.pt'))

        # 返回特征、生存时间和事件状态
        return [feature1, feature2, feature3], (row.time, row.event)
    
class slide_dataset_classify_one_size(object):
    '''
    Slide level dataset for survival analysis which returns:
    - feature matrices (h)
    - survival time
    - event status (1=event occurred, 0=censored)
    '''
    def __init__(self, df, encoder, pth):
        self.df = df
        self.pth = pth
        self.encoder = encoder
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # feature = torch.load(os.path.join(self.pth, row.pt + '_20.pt'))
        feature = torch.load(os.path.join(self.pth, row.pt + '.pt'))

        # 返回特征、生存时间和事件状态
        # return feature, (row.time, row.event)
        return feature, row.target


class slide_dataset_classify_multi_one_size(object):
    '''
    Multi-encoder slide level dataset for classify
    '''
    def __init__(self, df, pth):
        self.df = df
        self.pth = pth
        self.encoder = ['phikon', 'designeduni_freeze', 'exaonepath', 'Hibou_b']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]

        featres = []
        for enc in self.encoder:
            # feature = torch.load(os.path.join(self.pth, enc, 'pt_files', row.pt + '_20.pt'))
            feature = torch.load(os.path.join(self.pth, enc, 'pt_files', row.pt + '.pt'))

            # 统一特征维度
            if feature.shape[1] != 1536:
                feature = torch.nn.functional.interpolate(feature.unsqueeze(0), size=(1536,), mode='linear').squeeze(0)
            
            featres.append(feature)
        
        data = torch.stack(featres, dim=0)  # [num_encoder, N, 1536]

        # 返回多编码器特征、生存时间和事件状态
        return data, row.target


def get_datasets(encoder=''):
    """获取分类数据集"""
    base_path = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output'
    df = pd.read_csv(os.path.join(base_path, encoder+'.csv'))
    
    # 确保必要的列存在
    assert all(col in df.columns for col in ['time', 'event']), "CSV文件必须包含time和event列"
    
    # 分割训练集和验证集
    df_train = df[df.mccvsplit==0].reset_index(drop=True).drop(columns=['mccvsplit'])
    df_val = df[df.mccvsplit==1].reset_index(drop=True).drop(columns=['mccvsplit'])
    
    # 创建数据集对象
    dset_train = slide_dataset_survival(df_train, encoder=encoder, pth=os.path.join(base_path, encoder, 'pt_files'))
    dset_val = slide_dataset_survival(df_val, encoder=encoder, pth=os.path.join(base_path, encoder, 'pt_files'))
    
    return dset_train, dset_val

from sklearn.model_selection import KFold


def get_datasets_single_one_size(
    base_path = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/PAAD', 
    encoder='', 
    seed = 42
):
    """简化版：移除KFold，只返回一个完整的数据集（无训练/验证拆分）"""
    np.random.seed(seed)
    df = pd.read_csv(os.path.join(base_path, 'csv', 'designeduni_freeze.csv'))
    
    # 确保必要的列存在
    assert all(col in df.columns for col in ['pt', 'target']), "CSV文件必须包含pt和target列"
    
    # 使用全部数据创建数据集
    dset = slide_dataset_classify_one_size(df.reset_index(drop=True), encoder=encoder, pth=os.path.join(base_path, encoder, 'pt_files'))
    
    return dset

def get_datasets_multi_one_size(
    base_path = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/PAAD', 
    encoder='', 
    seed = 42
):
    """简化版：移除KFold，只返回一个完整的数据集（无训练/验证拆分）"""
    np.random.seed(seed)
    df = pd.read_csv(os.path.join(base_path, 'csv', 'designeduni_freeze.csv'))
    
    # 确保必要的列存在
    assert all(col in df.columns for col in ['pt', 'target']), "CSV文件必须包含pt和target列"
    
    # 使用全部数据创建数据集
    dset = slide_dataset_classify_multi_one_size(df.reset_index(drop=True), pth=base_path)
    
    return dset



def get_datasets_single_kfold_one_size(base_path = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/PAAD', encoder='', seed = 42, fold=None):
    """获取生存分析数据集"""
    np.random.seed(seed)
    df = pd.read_csv(os.path.join(base_path, 'csv', 'designeduni_freeze.csv'))
    
    # 确保必要的列存在
    assert all(col in df.columns for col in ['pt', 'target']), "CSV文件必须包含pt和target列"
    assert fold is not None, "必须指定fold参数"
    assert 0 <= fold < 5, "fold必须在0-4范围内"
    
    # 初始化5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    # 获取所有折的分割
    splits = list(kf.split(df))
    train_index, val_index = splits[fold]
    
    df_train = df.iloc[train_index].reset_index(drop=True)
    df_val = df.iloc[val_index].reset_index(drop=True)

    # 创建数据集对象
    dset_train = slide_dataset_classify_one_size(df_train, encoder=encoder, pth=os.path.join(base_path, encoder, 'pt_files'))
    dset_val = slide_dataset_classify_one_size(df_val, encoder=encoder, pth=os.path.join(base_path, encoder, 'pt_files'))
    
    return dset_train, dset_val


def get_datasets_multi_one_size_kfold(base_path='/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/PAAD', encoder='', seed=42, fold=None):
    # 设置随机数种子
    import numpy as np
    np.random.seed(seed)
    
    df = pd.read_csv(os.path.join(base_path, 'csv', 'designeduni_freeze.csv'))
    
    # 确保必要的列存在
    assert all(col in df.columns for col in ['pt', 'target']), "CSV文件必须包含pt和target列"
    assert fold is not None, "必须指定fold参数"
    assert 0 <= fold < 5, "fold必须在0-4范围内"
    
    # 初始化5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    # 获取所有折的分割
    splits = list(kf.split(df))
    train_index, val_index = splits[fold]
    
    df_train = df.iloc[train_index].reset_index(drop=True)
    df_val = df.iloc[val_index].reset_index(drop=True)
    
    # 创建数据集对象
    dset_train = slide_dataset_classify_multi_one_size(df_train, pth=base_path)
    dset_val = slide_dataset_classify_multi_one_size(df_val, pth=base_path)
    
    return dset_train, dset_val