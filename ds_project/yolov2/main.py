import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import yaml
from torch.utils.data import DataLoader, Dataset
import cv2
import matplotlib.pyplot as plt

from datasets import EPFLDataset
from transforms import Resize, Compose, HSVAdjust, VerticalFlip, Crop
from loader import custom_collate_fn
from train import train

from lightnet.models import YoloFusion
from rgb_yolo import Yolo
from losses import YoloLoss

def main():
    gt = "../data/relabeled.yaml"

    with open(gt, 'r') as f:
        df = (yaml.load(f, Loader=yaml.FullLoader))
    df_lab = {}
    for key, value in df.items():
        if 'epfl_lab' in key:
            df_lab[key] = value


    bboxes_by_id = {}
    for key, value in df_lab.items():
        bbox = []
        if 'person' in value:
            for obj in value['person']:
                bbox.append(obj['coords'])
        bboxes_by_id[int(key[-6:])] = bbox

    BATCH_SIZE = 8
    EPOCHS = 7
    IMG_SHAPE = (424, 512)
    FUSE = False
    imgs_path = "../data/epfl_lab/"

    aug = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(IMG_SHAPE)])

    dataset = EPFLDataset(imgs_path, bboxes_by_id)
    loader = DataLoader(dataset, BATCH_SIZE, drop_last=True, shuffle=True, collate_fn=custom_collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    num_classes = 1

    if FUSE:
        model = YoloFusion(num_classes, fuse_layer=18)
    else:
        model = Yolo(num_classes)

    model = model.to(device)

    reduction = 32
    criterion = YoloLoss(num_classes, model.anchors, device, reduction)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.0005)

    model.train()
    best_loss = 1e10
    best_epoch = 0

    train(loader, optimizer, model, device, EPOCHS, criterion)

    torch.save(model, './model.pt')

if __name__ == '__main__':
    main()