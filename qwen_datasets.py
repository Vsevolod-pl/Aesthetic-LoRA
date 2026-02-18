import os
import torch
import random
import torchvision
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as transforms


def load_img2batchfold(path2dataset):
    img2batchfold = dict()
    for p in Path(path2dataset).rglob('*'):
        img2batchfold[p.name] = str(p.parent.parent.resolve())
    return img2batchfold


def transform_PIL_adobe5k(img, target_shape=(504, 308)):
    w, h = img.size
    if h == 1000:
        img = img.transpose(method=Image.Transpose.ROTATE_90)
        w, h = h, w
    if h >= 600:
        to_crop = (h-600)//2
        img = img.crop((0, to_crop+h%2, w, h-to_crop))
    else:
        img = img.resize((1000, 600))
    img = img.resize(target_shape)
    return img


class Adobe5kDataset(Dataset):
    def __init__(self, markup, transform=None, path2batchfold=None, randomise=False,
                 preload=False, preload_downsample=None, device='cpu', tqdm=lambda a:a,
                 augmentation=None):
        self.markup = markup
        self.transform = transform
        self.img2batchfold = load_img2batchfold(path2batchfold)
        self.randomise = randomise
        self.preload = preload
        self.device = device
        self.preload_downsample = preload_downsample
        self.augmentation = augmentation

        if preload:
            img_paths = set()
            for idx in range(len(self.markup)):
                img_name = self.markup.name[idx]
                img_paths.add(f'{self.img2batchfold[img_name]}/{self.markup.left[idx]}/{img_name}')
                img_paths.add(f'{self.img2batchfold[img_name]}/{self.markup.right[idx]}/{img_name}')
            self.path2img = dict()
            for i, img_path in tqdm(list(enumerate(img_paths))):
                with Image.open(img_path) as img_stream:
                    img = img_stream.convert("RGB").copy()
                if self.preload_downsample is not None:
                    img = img.resize(self.preload_downsample)
                self.path2img[img_path] = img

    def __len__(self):
        return len(self.markup)

    def __getitem__(self, idx):
        img_name = self.markup.name[idx]
        img1_path = f'{self.img2batchfold[img_name]}/{self.markup.left[idx]}/{img_name}'
        img2_path = f'{self.img2batchfold[img_name]}/{self.markup.right[idx]}/{img_name}'
        label = self.markup.mos[idx]
        if self.randomise:
            if random.randint(0, 1):
                img1_path, img2_path = img2_path, img1_path
                label = 1.- label

        if self.preload:
            img1 = self.path2img[img1_path]
            img2 = self.path2img[img2_path]
        else:
            with Image.open(img1_path) as img_stream:
                img1 = img_stream.convert("RGB").copy()
            with Image.open(img2_path) as img_stream:
                img2 = img_stream.convert("RGB").copy()

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.augmentation:
            img1, img2 = self.augmentation(img1, img2)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at these two images.\nImage 1: "},
                {"type": "image", "image": img1},
                {"type": "text", "text": "\nImage 2: "},
                {"type": "image", "image": img2},
                {"type": "text", "text": "\nCompare them aesthetically. Which one is better? Answer \"1\" or \"2\"."}
            ],
        }]
        
        return messages, label


class ProcessorCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, things):
        messages, labels = list(zip(*things))
        labels = torch.Tensor(labels)
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        return inputs, labels
