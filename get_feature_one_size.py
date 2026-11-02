import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com '
import h5py
import torch
from torch.utils.data import DataLoader
from datasets_custom.dataset_h5 import *
from datasets_custom.dataset_wsi import *
import argparse
from utils.utils import seed_torch, get_args
from utils.utils_earlystop import EarlyStopping_loss
from torch.nn import DataParallel
from utils.utils import get_model, get_optim, NativeScalerWithGradNormCount
from torch import nn
import time
from torch.cuda import OutOfMemoryError
from tqdm import tqdm
import openslide
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='tumor', choices=['tumor'])
parser.add_argument('--yml_opt_path', type=str, default='designedUNI_freeze')  # designedUNI_freeze, phikon, hibou_b, exaonepath
parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use for parallel processing')
args = parser.parse_args()

def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    file.close()
    
    return output_path

def compute_w_loader(output_path, loader, model, device, verbose=0):
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches'.format(len(loader)))
    
    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            img = data['img'].to(device)
            coord = data['coord'].numpy().astype(np.int32)

            features = model(img)

            features = features.cpu().numpy().astype(np.float32)


            asset_dict = {'features': features, 'coords': coord}
            save_hdf5(output_path, asset_dict, mode=mode)
            mode = 'a'
    return output_path

def process_bag(bag_candidate_idx, bags_dataset, args, model, device):
    slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
    bag_name = slide_id + '.h5'
    h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
    slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
    print('\nprogress: {}/{}'.format(bag_candidate_idx+1, len(bags_dataset)))
    print(slide_id)

    if args.no_auto_skip and slide_id + '.pt' in os.listdir(os.path.join(args.feat_dir, args.model_type, 'pt_files')):
        print('skipped {}'.format(slide_id))
        return

    output_path = os.path.join(args.feat_dir, args.model_type, 'h5_files', bag_name)
    time_start = time.time()
    wsi = openslide.open_slide(slide_file_path)

    dataset = Whole_Slide_Bag_MIL_FP_all_patch(file_path=h5_file_path, wsi=wsi)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    model.to(device)  # 确保模型在正确的设备上
    output_file_path = compute_w_loader(output_path, loader=loader, model=model, device=device, verbose=1)

    time_elapsed = time.time() - time_start
    print('\ncomputing features for {} took {} s'.format(output_path, time_elapsed))
    
    if not os.path.exists(output_file_path):
        print('output file not found: {}'.format(output_file_path))
        return

    with h5py.File(output_file_path, "r") as file:
        features = file['features'][:]

    file.close()

    features = torch.from_numpy(features)

    bag_base, _ = os.path.splitext(bag_name)
    torch.save(features, os.path.join(args.feat_dir, args.model_type, 'pt_files', bag_base + '.pt'))


def worker(bag_candidate_idx, bags_dataset, args, model, semaphore, device):
    with semaphore:
        process_bag(bag_candidate_idx, bags_dataset, args, model, device)

if __name__ == "__main__":
    args = get_args(args)
    seed_torch(args.seed)
    # view settings
    settings = {}
    for key, value in args.__dict__.items():
        settings[key] = value
    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    print("################# Settings Finish ###################")

    bar = args.bar

    # dataset
    print('init dataloader')
    bags_dataset = Dataset_All_Bags(args.csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, args.model_type, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, args.model_type, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, args.model_type, 'pt_files'))

    # model
    print('Init Model...')
    model = get_model(args)
    print('model', type(model), flush=True)
    gpu_ids = args.gpu_ids
    print('gpu_ids', gpu_ids, flush=True)
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    torch.cuda.set_device(gpu_ids[0])
    if len(gpu_ids) > 1:
        model = DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()

    total = len(bags_dataset)

    # 创建一个信号量，限制同时运行的线程数量
    semaphore = threading.Semaphore(4)

    threads = []
    for bag_candidate_idx in range(total):
        device = torch.device(f'cuda:{gpu_ids[0]}')
        thread = threading.Thread(target=worker, args=(bag_candidate_idx, bags_dataset, args, model, semaphore, device))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()