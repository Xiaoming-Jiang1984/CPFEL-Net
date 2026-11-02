import os
import numpy as np
import pandas as pd
import torch

def get_datasets(base_path='/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/PAAD', encoder=''):
    # Load slide data
    csv_pth = os.path.join(base_path, 'csv')
    df = pd.read_csv(os.path.join(csv_pth, encoder+'.csv'))
    # Split into train and val
    df_train = df[df.mccvsplit==0].reset_index(drop=True).drop(columns=['mccvsplit'])
    df_val = df[df.mccvsplit==1].reset_index(drop=True).drop(columns=['mccvsplit'])
    # Create my loader objects
    dset_train = slide_dataset_classification(df_train, encoder=encoder, pth=os.path.join(base_path, encoder, 'pt_files'))
    dset_val = slide_dataset_classification(df_val, encoder=encoder, pth=os.path.join(base_path, encoder, 'pt_files'))
    return dset_train, dset_val

def get_datasets_multi(encoder=''):
    # Load slide data
    base_path = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/PAAD'
    df = pd.read_csv(os.path.join(base_path, 'designeduni_freeze.csv'))
    # Split into train and val
    df_train = df[df.mccvsplit==0].reset_index(drop=True).drop(columns=['mccvsplit'])
    df_val = df[df.mccvsplit==1].reset_index(drop=True).drop(columns=['mccvsplit'])
    # Create my loader objects
    dset_train = slide_dataset_classification_multi(df_train, pth=base_path)
    dset_val = slide_dataset_classification_multi(df_val, pth=base_path)
    return dset_train, dset_val


class slide_dataset_classification(object):
    '''
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    '''
    def __init__(self, df, encoder, pth):
        self.df = df
        self.pth = pth
        self.encoder = encoder
    
    def __len__(self):
        # number of slides
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        feature1 = torch.load(os.path.join(self.pth, row.pt + '_5.pt'))
        feature2 = torch.load(os.path.join(self.pth, row.pt + '_10.pt'))
        feature3 = torch.load(os.path.join(self.pth, row.pt + '_20.pt'))

        # get the target
        return [feature1, feature2, feature3], row.target


class slide_dataset_classification_multi(object):
    '''
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    '''
    def __init__(self, df, pth):
        self.df = df
        self.pth = pth
        self.encoder = ['phikon', 'designeduni_freeze', 'exaonepath', 'Prov_GigaPath', 'H_optimus_0', 'Hibou_b']
    
    def __len__(self):
        # number of slides
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        features5 = []
        features10 = []
        features20 = []
        for enc in self.encoder:
            feature5 = torch.load(os.path.join(self.pth, enc, 'pt_files', row.pt + '_5.pt'))
            feature10 = torch.load(os.path.join(self.pth, enc, 'pt_files', row.pt + '_10.pt'))
            feature20 = torch.load(os.path.join(self.pth, enc, 'pt_files', row.pt + '_20.pt'))

            # 将所有的特征维度调整为相同的维度
            if feature5.shape[1] != 1536:
                feature5 = torch.nn.functional.interpolate(feature5.unsqueeze(0), size=(1536,), mode='linear').squeeze(0)
            if feature10.shape[1] != 1536:
                feature10 = torch.nn.functional.interpolate(feature10.unsqueeze(0), size=(1536,), mode='linear').squeeze(0)
            if feature20.shape[1] != 1536:
                feature20 = torch.nn.functional.interpolate(feature20.unsqueeze(0), size=(1536,), mode='linear').squeeze(0)

            features5.append(feature5)
            features10.append(feature10)
            features20.append(feature20)
            
        data = [features5, features10, features20]

        return data, row.target
