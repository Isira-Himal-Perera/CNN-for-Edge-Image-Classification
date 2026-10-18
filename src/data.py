"""Consistent 70/15/15 split for both the offline Digits dataset and Beans."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
SIZE = 32  # always <=64; same resolution for every model
CLASSES_DIGITS = [str(k) for k in range(10)]
CLASSES_BEANS = ['angular_leaf_spot', 'bean_rust', 'healthy']


def _stratified_split(labels, seed=SEED):
    ids = np.arange(len(labels))
    tr, rem = train_test_split(ids, test_size=0.30, random_state=seed, stratify=labels)
    va, te = train_test_split(rem, test_size=0.50, random_state=seed, stratify=np.asarray(labels)[rem])
    return {'train': tr, 'val': va, 'test': te}


def load_offline_digits(image_size=SIZE):
    """UCI optical handwritten digits (scikit-learn copy): 1797 images of 8x8 pixels.
    RGB channels are repeated for equal model interfaces; images are resized to 32x32.
    """
    ds = load_digits()
    x = torch.tensor(ds.images, dtype=torch.float32)[:, None] / 16.0
    x = F.interpolate(x, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x = x.expand(-1, 3, -1, -1).contiguous()
    labels = ds.target.astype(np.int64)
    idx = _stratified_split(labels)
    splits = {k: (x[v], torch.from_numpy(labels[v])) for k, v in idx.items()}
    return splits, CLASSES_DIGITS, {'dataset': 'UCI Optical Recognition of Handwritten Digits via scikit-learn', 'original_pixels': '8x8 grayscale', 'input_pixels': f'{image_size}x{image_size} RGB (three copies of grayscale)', 'samples': len(labels), 'split_indices': {k: v.tolist() for k,v in idx.items()}}


def load_beans(image_size=SIZE):
    """Requires internet access and the datasets and Pillow packages."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('For Beans install datasets and Pillow: pip install datasets pillow') from exc
    from PIL import Image
    raw = load_dataset('AI-Lab-Makerere/beans')
    # Recombine the original splits before making the 70/15/15 division.
    imgs = []
    labs = []
    for splitname in ['train','validation','test']:
        for item in raw[splitname]:
            im = item['image'].convert('RGB').resize((image_size,image_size), Image.Resampling.BILINEAR)
            imgs.append(np.asarray(im,dtype=np.uint8))
            labs.append(int(item['labels']))
    labels = np.array(labs, dtype=np.int64)
    idx = _stratified_split(labels)
    all_x = torch.from_numpy(np.stack(imgs)).permute(0,3,1,2).float().div_(255.0)
    splits = {k:(all_x[v],torch.from_numpy(labels[v])) for k,v in idx.items()}
    return splits, CLASSES_BEANS, {'dataset':'Makerere AI Lab Beans', 'original_pixels':'500x500 RGB', 'input_pixels':f'{image_size}x{image_size} RGB', 'samples':len(labels), 'split_indices':{k:v.tolist() for k,v in idx.items()}}


def get_data(which='digits', size=SIZE):
    if which == 'digits':return load_offline_digits(size)
    if which == 'beans':return load_beans(size)
    raise ValueError(which)


def make_loaders(splits,batch_size=64,seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    return {k: DataLoader(TensorDataset(*v),batch_size=batch_size,shuffle=(k=='train'),generator=gen if k=='train' else None,num_workers=0) for k,v in splits.items()}
