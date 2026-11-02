from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pandas as pd
import torch
import h5py
import os
from sklearn.model_selection import StratifiedKFold

def get_transformations(target_img_size=224):
    tfms = []

    if target_img_size > 0:
        tfms.append(transforms.Resize(target_img_size))
		
    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    tfms = transforms.Compose(tfms)
    return tfms


class Dataset_All_Bags(Dataset):
	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]


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
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        img = self.roi_transforms(img)

        return {'img': img, 'coord': coord}
	
class WSIFeatureBagsDataset(Dataset):
    """
    Dataset for pre-extracted WSI bag features (.pt files) for ABMIL.
    Each .pt file corresponds to one WSI (bag).
    """

    def __init__(self, csv_path, feat_dir, model_type, slide_ext=".pt", transform=None):
        """
        Args:
            csv_path: Path to CSV file with ['slide_id', 'label'] columns
            feat_dir: Root directory for features
            model_type: subfolder for model type
            slide_ext: feature file extension (default '.pt')
            transform: optional transform
        """
        self.csv = pd.read_csv(csv_path)
        self.feat_dir = feat_dir  # os.path.join(feat_dir, model_type, "pt_files")
        self.slide_ext = slide_ext
        self.transform = transform
        self.model_type = model_type

        self.slide_ids = self.csv['slide_id'].tolist()
        self.labels = self.csv['label'].tolist()

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]
        if label == 0:
            dir_name = 'normal'
        elif label == 1:
            dir_name = 'tumor'
        else:
            raise ValueError(f"Unknown label {label} for slide {slide_id}")
        
        feat_path = os.path.join(self.feat_dir, dir_name, self.model_type, "pt_files", slide_id + self.slide_ext)
        features = torch.load(feat_path)  # shape: [num_patches, feature_dim]

        if self.transform:
            features = self.transform(features)

        return features, torch.tensor(label, dtype=torch.long)


def get_5fold_loaders(csv_path, feat_dir, model_type, batch_size=1, num_workers=0, seed=42):
    """
    Create 5-Fold DataLoaders for ABMIL training/validation.
    Returns a generator: (train_loader, val_loader) for each fold.
    """
    full_dataset = WSIFeatureBagsDataset(csv_path, feat_dir, model_type)
    labels = full_dataset.labels

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset.slide_ids, labels)):
        print(f"\n===== Fold {fold+1}/5 =====")
        print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        yield train_loader, val_loader

def get_test_loader(csv_path, feat_dir, model_type, batch_size=1, num_workers=0):
    """
    Build DataLoader for external test set.
    Args:
        csv_path: Path to external test CSV file
        feat_dir: Feature directory
        model_type: Model type subfolder
        batch_size: batch size (default 1 for MIL)
        num_workers: num_workers for DataLoader
    Returns:
        test_loader
    """
    test_dataset = WSIFeatureBagsDataset(csv_path, feat_dir, model_type)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_loader