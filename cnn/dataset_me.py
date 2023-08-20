from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch


class CatDogDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.root_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)

# Define Dataset
# class CatDogDataset(Dataset):
    
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         img_id = self.annotations.iloc[index, 0]
#         img_path = os.path.join(self.root_dir, img_id)
#         image = io.imread(img_path)
#         image = Image.fromarray(image)
#         y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

#         if self.transform:
#             image = self.transform(image)

#         return (image, y_label)
