import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import decode_image

from utils.load_config import load_config

LABEL_MAP = {
    "RBC": 1,   # Red Blood Cell
    "WBC": 2,   # White Blood Cell
    "Platelets": 3  # Platelets
}

def parse_annotation_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annos = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        label = LABEL_MAP.get(name, -1)
        if label == -1: 
            continue
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        annos.append({"boxes":[xmin,ymin,xmax,ymax], "labels":label})
    return annos

class BCCD_DATASET(Dataset):
    def __init__(self, root='BCCD_Dataset', mode='train', transform=None):
        # check mode
        if mode not in ['train','val','test']:
            raise ValueError("Mode should be 'train', 'val' or 'test'")
        
        self.transform = transform
        self.img_dir = os.path.join(root, 'BCCD', 'JPEGImages')
        self.ann_dir = os.path.join(root, 'BCCD', 'Annotations')
        id_file = os.path.join(root, 'BCCD', 'ImageSets', 'Main', f'{mode}.txt')
        with open(id_file, 'r') as f:
            self.ids = [x.strip() for x in f.readlines()]
        
        # cache annotations
        self.ann = {}
        for img_id in self.ids:
            anno = parse_annotation_xml(os.path.join(self.ann_dir, img_id + '.xml'))
            self.ann[img_id] = anno

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        xml_path = os.path.join(self.ann_dir, img_id + '.xml')

        # read image as RGB np.uint8 (HWC)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # collect boxes/labels
        raw = self.ann[img_id]
        boxes = []
        labels = []
        for r in raw:
            xmin, ymin, xmax, ymax = r["boxes"]
            # clamp + fix bad order if any
            xmin = max(0.0, min(xmin, w-1))
            ymin = max(0.0, min(ymin, h-1))
            xmax = max(0.0, min(xmax, w-1))
            ymax = max(0.0, min(ymax, h-1))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(r["labels"]))

        # Albumentations expects uint8 HWC + list bboxes
        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed["image"]              # tensor (C,H,W) [0,1]
            boxes = transformed["bboxes"]           # list of tuples
            labels = transformed["labels"]          # list of ints

        # reshape to CHW tensor [0,1]
        img = img.float()/255.0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }
        return img, target