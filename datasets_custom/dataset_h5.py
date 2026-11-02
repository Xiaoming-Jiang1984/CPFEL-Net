import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py


def get_transformations(target_img_size=224):
    tfms = []

    if target_img_size > 0:
        tfms.append(transforms.Resize(target_img_size))
		
    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    tfms = transforms.Compose(tfms)
    return tfms

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):
	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]


class Whole_Slide_Bag_MIL_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=get_transformations(target_img_size=224)):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms
		self.file_path = file_path
		self.is_valid = True

		with h5py.File(self.file_path, "r") as f:
			dset_all = f['coords']
			# 随机取30%的coords,seed取42以确保每次运行结果一致
			np.random.seed(42)
			if len(dset_all) < 2000:
				# 如果数据量小于1000，则全部使用
				# 创建一个txt，记录数量不足的名字
				with open('dataset_size_warning.txt', 'a') as f:
					f.write(self.file_path)

				print(f'Warning: dataset size {len(dset_all)} is less than 2000, using all data.')

				self.is_valid = False  # 设置标志为 False
				self.dset = []  # 设置为空数据集
				self.patch_level = None  # 设置为 None
				self.patch_size = None  # 设置为 None
				self.length = 0  # 设置长度为 0

			else:
				# number = int(len(dset_all) * 0.3)
				number = 2000
				idx = np.random.choice(len(dset_all), number, replace=False)
				idx.sort()  # 关键：保证索引递增
				dset = dset_all[idx]
				self.dset = dset	 
				self.patch_level = f['coords'].attrs['patch_level']
				self.patch_size = f['coords'].attrs['patch_size']
				self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		# dset = self.dset
		for name, value in dset.attrs.items():
			print(name, value)

		# print('\nfeature extraction settings')
		# print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		# with h5py.File(self.file_path,'r') as hdf5_file:
			# coord = hdf5_file['coords'][idx]
		if not self.is_valid:
			raise IndexError("Dataset is invalid due to insufficient data.")
		
		coord = self.dset[idx]
		img20 = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		# 在coord的向左和向上各偏移256个像素
		coord10 = (round(coord[0] - self.patch_size/2), round(coord[1] - self.patch_size/2))
		patch_size10 = 2 * self.patch_size
		img10 = self.wsi.read_region(coord10, self.patch_level, (patch_size10, patch_size10)).convert('RGB') 
		coord5 = (round(coord10[0] - patch_size10/2), round(coord10[1] - patch_size10/2))
		patch_size5 = 2 * patch_size10
		img5 = self.wsi.read_region(coord5, self.patch_level, (patch_size5, patch_size5)).convert('RGB')

		img20 = self.roi_transforms(img20)
		img10 = self.roi_transforms(img10)
		img5 = self.roi_transforms(img5)
		return {'img20': img20, 'img10': img10, 'img5': img5, 'coord': coord}


class Whole_Slide_Bag_MIL_FP_all_patch(Dataset):
    def __init__(self, file_path, wsi, img_transforms=get_transformations(target_img_size=224)):
        self.wsi = wsi
        self.roi_transforms = img_transforms
        self.file_path = file_path
        self.is_valid = True

        with h5py.File(self.file_path, "r") as f:
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(f['coords'])

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        with h5py.File(self.file_path, "r") as hdf5_file:
            dset = hdf5_file['coords']
            for name, value in dset.attrs.items():
                print(name, value)
        # print('\nfeature extraction settings')
        # print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        if not self.is_valid:
            raise IndexError("Dataset is invalid due to insufficient data.")
        with h5py.File(self.file_path, "r") as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img20 = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        coord10 = (round(coord[0] - self.patch_size/2), round(coord[1] - self.patch_size/2))
        patch_size10 = 2 * self.patch_size
        img10 = self.wsi.read_region(coord10, self.patch_level, (patch_size10, patch_size10)).convert('RGB') 
        coord5 = (round(coord10[0] - patch_size10/2), round(coord10[1] - patch_size10/2))
        patch_size5 = 2 * patch_size10
        img5 = self.wsi.read_region(coord5, self.patch_level, (patch_size5, patch_size5)).convert('RGB')

        img20 = self.roi_transforms(img20)
        img10 = self.roi_transforms(img10)
        img5 = self.roi_transforms(img5)
        return {'img20': img20, 'img10': img10, 'img5': img5, 'coord': coord}