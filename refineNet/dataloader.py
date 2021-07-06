## Contains DataLoader for Cityscapes

import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import random

class MyDataset(data.Dataset):
    def __init__(self, list_IDs, labels, train=True):
        
        self.labels = labels
        self.list_IDs = list_IDs
        self.train = train

    def transform(self, image, mask, depth):
    
        # Resize
        resize = transforms.Resize(size=(256, 512))
        image = resize(image)
        mask = resize(mask)
        depth = resize(depth)

        # Random horizontal flipping
        if random.random() > 0.5 and train:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Transform to tensor
        t = transforms.ToTensor()
        image = t(image)
        mask = t(mask)
        depth = t(depth)

        # Normalise
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        normalize1 = transforms.Normalize([0.491], [0.2023])
        image = normalize(image)
        depth = normalize1(depth)

        image = torch.cat((image, depth), 0)
    
        return image, mask

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        new_path = '/content/drive/My Drive/depth_city/'
        image = Image.open(ID)
        mask = Image.open(self.labels[ID])
        depth = Image.open(new_path + ID[33:])
        # ColorJitter
        color = transforms.ColorJitter(brightness=.05, contrast=.05, hue=.05, saturation=.05)
        if train:
            image = color(image)
        w, h = image.size
        image = image.crop((0, 0, w, h-20))
        mask = mask.crop((0, 0, w, h-20))
        x, y = self.transform(image, mask, depth)
        return x, y

    def __len__(self):
        return len(self.list_IDs)