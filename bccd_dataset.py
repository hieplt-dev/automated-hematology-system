import os
import torch
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import pdb

def parse_annotation_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    annotations = []

    # size_element = root.find('size')
    # width = int(size_element.find('width').text)
    # height = int(size_element.find('height').text)

    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        annotations.append({
            'boxes': [xmin, ymin, xmax, ymax],
            'labels': label
        })
    
    return annotations

class BCCD_DATASET(Dataset):
    def __init__(self, root='BCCD_Dataset', mode='train', transform=None):
        # check mode
        if mode not in ['train', 'test', 'val']:
            raise ValueError("Mode must be 'train', 'test' or 'val'")
        
        self.transform = transform
        self.img_path = os.path.join(root, 'BCCD', 'JPEGImages')
        
        # set dir paths
        self.img_name_dir = os.path.join(root, 'BCCD', 'ImageSets', 'Main')
        self.anno_name_dir = os.path.join(root, 'BCCD', 'Annotations')
        
        with open(os.path.join(self.img_name_dir, f'{mode}.txt'), 'r') as f:
            self.name_img_ls = [line.strip() for line in f.readlines()]
            

    def __len__(self):
        return len(self.name_img_ls)

    def __getitem__(self, idx):
        img_file = os.path.join(self.img_path, self.name_img_ls[idx] + '.jpg')
        anno_file = os.path.join(self.anno_name_dir, self.name_img_ls[idx] + '.xml')
        
        # Check if files exist
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image file {img_file} does not exist.")
        if not os.path.exists(anno_file):
            raise FileNotFoundError(f"Annotation file {anno_file} does not exist.")
        
        # Load image
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        if self.transform:
            img = self.transform(img)
        
        # Load annotations
        anno = parse_annotation_xml(anno_file)
        
        # pdb.set_trace()  # Debugging point
        
        
        return img, anno
    
if __name__ == "__main__":
    dataset = BCCD_DATASET(root='BCCD_Dataset', mode='train')
    img, anno = dataset[0]
    print(img.shape)
    print(anno)  # Print first annotation for debugging
    