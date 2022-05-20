import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

import torchvision.transforms as T
from pathlib import Path

def get_transform(args):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((256, 192)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((256, 192)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
