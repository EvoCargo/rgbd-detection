import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

class EPFLDataset_replace(Dataset):
    def __init__(self, path, bbox, transforms=None, replace=False):
        self.transforms = transforms
        self.img_path = path
        self.bbox = bbox
        self.replace = replace

    
    def __getitem__(self, idx):
        image_path = os.path.join(os.path.join(self.img_path, 'images') + "/rgb"+ str(idx).zfill(6) + ".png")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = self.bbox[idx]
        objects = []
        for box in boxes:
            label = 0
            objects.append([box[0], box[1], box[0] + box[2], box[1] + box[3], label])

        if self.replace is not None:
            assert self.replace in range(3), 'Replace index should be in [0, 2]'
            depth_path = os.path.join(os.path.join(self.img_path, 'depth') + "/depth"+ str(idx).zfill(6) + ".png")
            depth = cv2.imread(depth_path)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            image[..., self.replace] = depth

        if self.transforms:
            image, objects = self.transforms((image, objects))

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)


    def __len__(self):
        return len(os.listdir(os.path.join(self.img_path, 'images')))