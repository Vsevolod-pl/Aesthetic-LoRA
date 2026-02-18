import gc
import os
import torch
import mlflow
import random
import tempfile
import argparse
import traceback
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

import sys
sys.path.append('../..')
sys.path.append('../../..')

from VLM_models import models
from utils import load_json
from NN_train_funcs import train_loop
from NN_Tester import Tester
from shops import optimizers as opt_shop
from shops import schedulers as sched_shop
from shops import losses as loss_shop
from aug_shop import augmentations_set
from qwen_datasets import Adobe5kDataset, transform_PIL_adobe5k, ProcessorCollator


def create_paired_aug(aug):
    def paired_aug(img1, img2):
        return aug(torch.stack([img1, img2]))
    return paired_aug


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_run(params, train_set, test_set, device):
    MODEL_NAME = params['BASE_MODEL_NAME']
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    col = ProcessorCollator(processor)
    train_loader = DataLoader(train_set, batch_size=params["train_batch_size"], shuffle=True, collate_fn=col)
    tester = Tester(test_set, device=device, tqdm=tqdm, bs=params["test_batch_size"], preload=False, dataloader_kwargs={'collate_fn':col})

    mlflow.set_tracking_uri(uri="http://10.0.111.233:6060")

    # Create a new MLflow Experiment
    mlflow.set_experiment(params['experiment_name'])

    with mlflow.start_run(run_name=params['run_name']):
        try:
            mlflow.log_params(params)
            
            model = models[params['model_name']](MODEL_NAME=MODEL_NAME, lora_config_params=params.get('lora_config_params'), device=device)
            model.to(device);
            criterion = loss_shop[params['criterion']]()
            optimizer = opt_shop[params['optimizer_name']](model.parameters(), **params['optimizer_params'])
            sched = sched_shop[params['scheduler_name']](optimizer, **params['scheduler_params'])
        
            hist = train_loop(params['run_name'],
                              train_loader,
                              tester,
                              criterion,
                              model,
                              optimizer,
                              sched,
                              device,
                              params['num_epochs'],
                              params['save_freq'],
                              path2save='./weights/',
                              logging_func=mlflow.log_metric,
                              trg=trange)
        except Exception as e:
            err_formated = traceback.format_exc()
            mlflow.log_param("error:", err_formated)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log', prefix=params['run_name'], dir='./error_logs/') as temp_file:
                temp_file.write(err_formated)
                temp_path = temp_file.name
                mlflow.log_artifact(temp_path, "error_logs")
            raise

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(description="Run N training runs with different seeds")
    parser.add_argument('--config', type=Path, required=True, help='Path to JSON config file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('-N', '--n_iters', dest='N', type=int, default=10, help='number of reruns')
    parser.add_argument('-G', '--preload_on_gpu', action='store_true', help='enable preload to gpu')
    parser.add_argument('-P', '--preload', action='store_true', help='enable preload')
    parser.add_argument('-d', '--dataset_path', type=Path, default=Path('/mnt/nuberg/datasets/IQA/5k'))
    parser.add_argument('--debug', action='store_true', help='Debug mode: cut the dataset to 100 samples')

    args = parser.parse_args()
    params = load_json(args.config)
    downsample = params.get("downsample", None)
    if params["dataset_version"] == 'current_experiment':
        params["dataset_version"] = load_json('../../../named_versions.json')['current_experiment']

    aug_list = []
    for aug_name, aug_params in params.get("augmentations", {}).items():
        aug_list.append(augmentations_set[aug_name](**aug_params))
    aug = None
    if aug_list:
        raise NotImplementedError("Sorry, no augmentations for Qwen yet")
        aug = create_paired_aug(transforms.Compose(aug_list))

    train_markup = pd.read_csv(f'../../../versioned_datasets/train_{params["dataset_version"]}.csv')
    test_markup = pd.read_csv(f'../../../versioned_datasets/test_{params["dataset_version"]}.csv')
    if args.debug:
        train_markup = train_markup[:50]
        test_markup = test_markup[:20]
    transform = None
    
    if downsample:
        def resize_adobe5k(img, resize=downsample):
            return transform_PIL_adobe5k(img)
        transform = resize_adobe5k
    dataset_device = 'cpu'
    preload_transform = None
    if args.preload_on_gpu:
        raise NotImplementedError("Sorry, no preload on GPU for Qwen yet")
        preload_transform = transforms.Compose([
            transform_adobe5k,
        ])
        transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if downsample:
            preload_transform = transforms.Compose([
                transform_adobe5k,
                transforms.Resize(downsample),
            ])
        dataset_device = args.device
    train_set = Adobe5kDataset(train_markup, transform=transform, randomise=True, path2batchfold=args.dataset_path, preload=args.preload, tqdm=tqdm, device=dataset_device, preload_downsample=downsample, augmentation=aug)
    test_set = Adobe5kDataset(test_markup, transform=transform, path2batchfold=args.dataset_path, preload=args.preload, tqdm=tqdm, device=dataset_device, preload_downsample=downsample)
    
    device = args.device
    N = args.N

    for i in range(3407, 3407+N):
        params = load_json(args.config)
        if params["dataset_version"] == 'current_experiment':
            params["dataset_version"] = load_json('../../../named_versions.json')['current_experiment']
        params['run_name'] = params['dataset_version']+'_'+params['model_name'] + f'_{i}' + params.get('run_name', '')
        set_seed(i)
        train_run(params, train_set, test_set, device)
        gc.collect()
        torch.cuda.empty_cache()
