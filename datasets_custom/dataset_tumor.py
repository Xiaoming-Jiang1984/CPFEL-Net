# tumor 其他 二分类
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from datasets_custom.transforms_utils import *
import glob
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch

def get_transforms(img_size=224, mode='train'):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if mode == 'train':
        my_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.36, 1.0), interpolation=InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )
    else:
        my_transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )
    return my_transforms


class tumor_classify_dataset(Dataset):
    def __init__(self,
                 mode='train',
                 seed=42,
                 in_memory=False,
                 bar=False
                 ):
        super().__init__()
        self.in_memory = in_memory
        print('mode', mode)
        random.seed(seed)
        img_suffix_list = ['.jpg', '.jpeg', '.png']
        tumor_root = [
            '/mnt/SSD_4T/admin103/ydw/UNI_classify/cut_patch/patch_tumor'
        ]  # tumor的wsi的patch
        other_root = [
            '/mnt/SSD_4T/admin103/ydw/UNI_classify/cut_patch/patch_normal',
        ]  # 正常的wsi的patch
        tumor_pth = []
        for x in tumor_root:
            for p in img_suffix_list:
                tumor_pth += glob.glob(f"{x}/**/*{p}", recursive=True)
        other_pth = []
        for x in other_root:
            for p in img_suffix_list:
                other_pth += glob.glob(f"{x}/**/*{p}", recursive=True)

        # print(tumor_pth)
        # print(other_pth)
        # print(len(tumor_pth), len(other_pth))

        pth_all = tumor_pth + other_pth
        pth_all.sort()
        print('all patch num', len(pth_all))

        train_size = 0.6
        train_data, temp_data = train_test_split(pth_all, train_size=train_size, random_state=seed)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)
        print('train_data, val_data, test_data', len(train_data), len(val_data), len(test_data))
        if mode == 'train':
            self.pth_all = train_data
        elif mode == 'val':
            self.pth_all = val_data
        elif mode == 'test':
            self.pth_all = test_data
        else:
            raise NotImplementedError

        print(mode, 'patches', len(self.pth_all))
        # print(self.pth_all)
        self.n = len(self.pth_all)
        self.transforms = get_transforms(mode=mode)

        if self.in_memory:
            print('load data to memory', flush=True)
            self.all_imgs = []
            self.all_labels = []
            if bar:
                lstt = tqdm.tqdm(self.pth_all)
            else:
                lstt = self.pth_all
            for pth in lstt:
                img = Image.open(pth).convert('RGB')
                if 'tumor' in pth:
                    label = 1
                else:
                    label = 0

                self.all_imgs.append(img)
                self.all_labels.append(label)
        else:
            print('load data from disk', flush=True)
        print(mode, 'dataset init end', flush=True)

    def __getitem__(self, idx):
        idx = idx % self.n
        if self.in_memory:
            img = self.all_imgs[idx]
            label = self.all_labels[idx]
        else:
            pth = self.pth_all[idx]
            img = Image.open(pth).convert('RGB')
            # img = Image.open(pth)
            if 'tumor' in pth:
                label = 1
            else:
                label = 0

        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.pth_all)
    
class tumor_classify_dataset_my(Dataset):
    def __init__(self,
                 mode='train',
                 seed=42,
                 in_memory=False,
                 bar=False
                 ):
        super().__init__()
        self.in_memory = in_memory
        print('mode', mode)
        random.seed(seed)
        img_suffix_list = ['.jpg', '.jpeg', '.png']
        train_root = [
            '/mnt/SSD_4T/admin103/ydw/UNI_classify/cut_patch/train'
        ]  # tumor的wsi的patch
        val_root = [
            '/mnt/SSD_4T/admin103/ydw/UNI_classify/cut_patch/val'
        ]  # 正常的wsi的patch

        self.train = {}
        for x in train_root:
            for file in os.listdir(x):
                # 判断文件夹x是否是‘train’
                if file == 'tumor':
                    f_pth = os.path.join(x, file)
                    for p in img_suffix_list:
                        pth_train = glob.glob(f"{f_pth}/**/*{p}", recursive=True)
                    for pth in pth_train:
                        label = 1   
                        self.train[pth] = label

                elif file == 'normal':
                    f_pth = os.path.join(x, file)
                    for p in img_suffix_list:
                        pth_train = glob.glob(f"{f_pth}/**/*{p}", recursive=True)
                    for pth in pth_train:
                        label = 0    
                        self.train[pth] = label

                else:
                    raise NotImplementedError
        
        items = list(self.train.items())
        random.shuffle(items)
        self.train = dict(items)

        # val_dataset
        self.val = {}
        for x in val_root:
            for file in os.listdir(x):
                    # 判断文件夹x是否是‘val’
                    if file == 'tumor':
                        f_pth = os.path.join(x, file)
                        for p in img_suffix_list:
                            pth_train = glob.glob(f"{f_pth}/**/*{p}", recursive=True)
                        for pth in pth_train:
                            label = 1   
                            self.val[pth] = label

                    elif file == 'normal':
                        f_pth = os.path.join(x, file)
                        for p in img_suffix_list:
                            pth_train = glob.glob(f"{f_pth}/**/*{p}", recursive=True)
                        for pth in pth_train:
                            label = 0    
                            self.val[pth] = label

                    else:
                        raise NotImplementedError
            
        
        items = list(self.val.items())
        random.shuffle(items)
        self.val = dict(items)

        if mode == 'train':
            self.pth_all = self.train.keys()
        elif mode == 'val':
            self.pth_all = self.val.keys()
        elif mode == 'test':
            # self.pth_all = test_data
            print("test data not found")
        else:
            raise NotImplementedError
        
        print(mode, 'patches', len(self.pth_all))
        # print(self.pth_all)
        self.n = len(self.pth_all)
        self.transforms = get_transforms(mode=mode)

        if self.in_memory:
            print('load data to memory', flush=True)
            self.all_imgs = []
            self.all_labels = []

            if bar:
                lstt = tqdm.tqdm(self.pth_all)
            else:
                lstt = self.pth_all

            for pth in lstt:
                img = Image.open(pth).convert('RGB')
                if mode == 'train':
                    label = self.train[pth]
                elif mode == 'val':
                    label = self.val[pth]
                else:
                    raise NotImplementedError

                self.all_imgs.append(img)
                self.all_labels.append(label)
            
            print('all patch num', len(self.pth_all))
        print(mode, 'dataset init end', flush=True)
        

    def __getitem__(self, idx):
        idx = idx % self.n
        if self.in_memory:
            img = self.all_imgs[idx]
            label = self.all_labels[idx]
        else:
            pth = self.pth_all[idx]
            img = Image.open(pth).convert('RGB')
            # img = Image.open(pth)
            # if 'tumor' in pth:
            #     label = 1
            # else:
            #     label = 0
            label = self.train[pth]
        if(label != 0 and label != 1):
            print('label', label)
            print('pth', pth)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.pth_all)
    
# 测试集用的dataset
class tumor_classify_dataset_test(Dataset):
    def __init__(self,
                 test_name = '060-CI;327-B',
                 mode='test',
                 seed=42,
                 in_memory=False,
                 bar=False
                 ):
        super().__init__()
        self.in_memory = in_memory
        print('mode', mode)
        random.seed(seed)
        img_suffix_list = ['.jpg', '.jpeg', '.png']
        tumor_root = [
            '/mnt/SSD_4T/admin103/ydw/UNI_classify/cut_patch/test/'+ test_name+ '/tumor'
        ]  # tumor的wsi的patch
        other_root = [
            '/mnt/SSD_4T/admin103/ydw/UNI_classify/cut_patch/test/'+ test_name+ '/normal'
        ]  # 正常的wsi的patch
        tumor_pth = []
        for x in tumor_root:
            for p in img_suffix_list:
                tumor_pth += glob.glob(f"{x}/**/*{p}", recursive=True)
        other_pth = []
        for x in other_root:
            for p in img_suffix_list:
                other_pth += glob.glob(f"{x}/**/*{p}", recursive=True)

        # print(tumor_pth)
        # print(other_pth)
        # print(len(tumor_pth), len(other_pth))

        pth_all = tumor_pth + other_pth
        pth_all.sort()
        print('all patch num', len(pth_all))

        train_data = val_data = []
        test_data = pth_all

        if mode == 'train':
            self.pth_all = train_data
        elif mode == 'val':
            self.pth_all = val_data
        elif mode == 'test':
            self.pth_all = test_data
        else:
            raise NotImplementedError

        print(mode, 'patches', len(self.pth_all))
        # print(self.pth_all)
        self.n = len(self.pth_all)
        self.transforms = get_transforms(mode=mode)

        print('load data to memory', flush=True)
        self.all_imgs = []
        self.all_labels = []
        self.all_pths = []

        if bar:
                lstt = tqdm.tqdm(self.pth_all)
        else:
            lstt = self.pth_all
    
        for pth in lstt:
            img = Image.open(pth).convert('RGB')
            if 'tumor' in pth:
                label = 1
            else:
                label = 0

            self.all_imgs.append(img)
            self.all_labels.append(label)
            self.all_pths.append(pth)
                
        print(mode, 'dataset init end', flush=True)

    def __getitem__(self, idx):
        idx = idx % self.n
        if self.in_memory:
            img = self.all_imgs[idx]
            label = self.all_labels[idx]
        else:
            pth = self.pth_all[idx]
            img = Image.open(pth).convert('RGB')
            # img = Image.open(pth)
            if 'tumor' in pth:
                label = 1
            else:
                label = 0

        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.pth_all)

class tumor_classify_dataset_panda(Dataset):
    def __init__(self,
                 mode='train',
                 seed=42,
                 in_memory=False,
                 bar=False
                 ):
        super().__init__()
        self.in_memory = in_memory
        print('mode', mode)
        random.seed(seed)
        # img_suffix_list = ['.jpg', '.jpeg', '.png']
        tumor_root = [
            '/mnt/SSD_4T/admin103/ydw/UNI_classify/cut_patch/panda/tumor'
        ]  # tumor的wsi的patch
        other_root = [
            '/mnt/SSD_4T/admin103/ydw/UNI_classify/cut_patch/panda/normal',
        ]  # 正常的wsi的patch
        self.csv_panda = '/mnt/SSD_4T/admin103/ydw/Intraductal_cancer_data/panda/train.csv'
        wsi_tumor_pth = []
        for x in tumor_root:
            # for file in os.listdir(x):
            # x对应的每个文件夹就是一个wsi
            wsi_tumor_pth = glob.glob(f"{x}/*")
        wsi_normal_pth = []
        for x in other_root:
            wsi_normal_pth = glob.glob(f"{x}/*")

        # print(tumor_pth)
        # print(other_pth)
        print(len(wsi_tumor_pth), len(wsi_normal_pth))

        pth_all = wsi_tumor_pth + wsi_normal_pth
        pth_all.sort()
        print('all wsi num', len(pth_all))

        train_size = 0.6
        train_data, temp_data = train_test_split(pth_all, train_size=train_size, random_state=seed)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)
        print('train_data, val_data, test_data', len(train_data), len(val_data), len(test_data))
        if mode == 'train':
            self.pth_all = train_data
        elif mode == 'val':
            self.pth_all = val_data
        elif mode == 'test':
            self.pth_all = test_data
        else:
            raise NotImplementedError

        print(mode, 'wsi', len(self.pth_all))
        # print(self.pth_all)
        self.n = len(self.pth_all)
        self.transforms = get_transforms(mode=mode)

        # print('load data to memory', flush=True)
        print('load data from disk', flush=True)
        self.wsi_imgs = []
        self.wsi_labels = {}

        lstt = self.pth_all

        for pth in lstt:
            one_wsi_pth = []
            for p in os.listdir(pth):
                p_pth = os.path.join(pth, p)
                one_wsi_pth.append(p_pth)
        
            # 根据p找到在csv_panda中的label
            df = pd.read_csv(self.csv_panda)
            row = df[df['image_id'] == pth.split('/')[-1]]
            if not row.empty:
                one_wsi_label = row['isup_grade'].values[0]
            else:
                print('error: label not found')
                return

            self.wsi_imgs.append(one_wsi_pth) 
            self.wsi_labels[pth.split('/')[-1]] = one_wsi_label

        print(mode, 'dataset init end', flush=True)

    def __getitem__(self, idx):
        idx = idx % self.n

        one_wsi_pth = self.wsi_imgs[idx]
        one_wsi_imgs = []
        for patch in os.listdir(one_wsi_pth):
            patch_pth = os.path.join(one_wsi_pth, patch)
            img = Image.open(patch_pth).convert('RGB')
            
            if self.transforms:
                img = self.transforms(img)
                
            one_wsi_imgs.append(img)
        
        # 将所有patch 组合为一个batch
        img = torch.stack(one_wsi_imgs)

        label = self.wsi_labels[one_wsi_pth.split('/')[-1]]

        return img, label

    def __len__(self):
        return len(self.pth_all)
    

if __name__ == "__main__":
    dt = tumor_classify_dataset_panda()

    img, lb = dt[0]
    # print(img)
    # print(lb)
