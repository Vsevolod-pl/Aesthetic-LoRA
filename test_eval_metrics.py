import torch
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from transformers import AutoProcessor

from qwen_datasets import Adobe5kDataset, ProcessorCollator
from VLM_models import LoRA_Qwen
import sys
sys.path.append('../..')
sys.path.append('../../..')
from utils import load_json, store_results, eval_metrics
from NN_Tester import Tester

device = 'cuda:0'
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

def downsample_twice(img):
    w, h = img.size
    resize_factor = round(max(w, h) / 1000)
    # print(w, h, ': ', w//resize_factor, h//resize_factor)
    img = img.resize((w//resize_factor, h//resize_factor))
    return img
# test_markup = pd.read_csv('../../../versioned_datasets/test_memorably_interludes.csv')
# test_set = Adobe5kDataset(test_markup, path2batchfold='/mnt/nuberg/datasets/IQA/5k', preload=True, tqdm=tqdm, transform=downsample_twice)

test_markup = pd.read_csv('./ayy/Artem_lol.csv')
test_set = Adobe5kDataset(test_markup, path2batchfold='./ayy/', preload=True, tqdm=tqdm, transform=downsample_twice)

processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
col = ProcessorCollator(processor)
tester = Tester(test_set, device=device, tqdm=tqdm, bs=2, preload=False, dataloader_kwargs={'collate_fn':col})

classifier = LoRA_Qwen(device=device, MODEL_NAME=MODEL_NAME)
# MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# stdct = torch.load('./weights/memorably_interludes_LoRA_Qwen2_VL_3407_debug/memorably_interludes_LoRA_Qwen2_VL_3407_debug_6.pth', weights_only=False, map_location=device)
# classifier = LoRA_Qwen(device=device)
# classifier.load_state_dict(stdct['model'])

with torch.inference_mode():
    test_predict, test_lbls = tester.predict(classifier, tqdm=tqdm)
test_lbls = test_lbls.copy()
# np.savez_compressed('test_results.npz', test_predict=test_predict, test_lbls=test_lbls)

test_markup.left, test_markup.right = test_markup.right, test_markup.left
test_markup.mos = 1-test_markup.mos

with torch.inference_mode():
    test_predict_mirrored, test_lbls_mirrored = tester.predict(classifier, tqdm=tqdm)
# np.savez_compressed('test_results_mirrored.npz', test_predict=test_predict, test_lbls=test_lbls)
# test_predict = (test_predict + (1-test_predict_mirrored))/2
print(eval_metrics(test_predict, test_lbls))