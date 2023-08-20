import os
import pathlib
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
from PIL import Image
from skimage.color import rgba2rgb
from skimage.io import imread
import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from skimage.io import imread
from typing import List, Union
import json
from utils import read_json, map_class_to_int
import torchvision.transforms

# # Method 1- works 
# Iterate image in the root folder and convert each image to RGB format. 
# root = Path("/home/jingying/AIPython/data/headdata/input")
# inputs = [file for file in root.iterdir() if file.is_file()]
# inputs = inputs.sort()
# print(inputs)
# input_ID = inputs[1]
# print(input_ID)
# img = Image.open(input_ID).convert("RGB")
# print(img)
# img = torch.from_numpy(img).type(torch.float32)
# print(img)

# Method 2 - works
# Iterate image in the root folder and convert each image to RGB format. 
root = '/home/jingying/AIPython/data/headdata/input'
imgs = [image for image in sorted(os.listdir(root)) if image[-4:]=='.jpg']
# print(imgs)
img_name = imgs[0]
# print(img_name)
img_path = os.path.join(root, img_name)
# print(image_path)
img = Image.open(img_path).convert("RGB")
# print(img)
# sub_method_3
image = imread(img_path)
image = Image.fromarray(image)
# print(image)
# img = torch.from_numpy(image).type(torch.float32)
# img = torch.tensor(image).to(torch.float32)

img = torchvision.transforms.ToTensor(image)
print(img)



# # The below code works: read json file 
# root = '/home/jingying/AIPython/data/headdata'
# annotations = [file for file in sorted(os.listdir(os.path.join(root, "target")))]
# print(annotations)
# target_ID = annotations[0]
# print(target_ID)
# target_path = os.path.join(root, "target", target_ID)
# print(target_path)
# def read_json(path: pathlib.Path) -> dict:
#     with open(str(path), "r") as fp:  # fp is the file pointer
#         file = json.loads(s=fp.read())

#     return file


# y = read_json(target_path)
# print(y)
# print(y.keys())
# boxes = y['boxes']
# print(boxes)


# y1 = pd.read_json(target_path)
# # print(y1)

# with open(target_path) as f:
#     y2 = json.load(f)
#     print(y2)

# Read labels - below code not working - name '_' is not defined
# classes = [_, 'head','banana','orange']
# labels = []
# label_names = y["labels"]
# labels = labels.append(classes.index(label_names))
# print(labels)


# Read labels - method2
# mapping = {
#     'head': 1
# }

# if mapping:
#     labels = map_class_to_int(y["labels"], mapping=mapping)
# else:
#     labels = y["labels"]


# print(labels)

# Different methods to read json file. - code good
# # Read Json file.
# file_path ="/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/gqa/all_val_data.json"
# question_json = pd.read_json(file_path)
# question_json[["questions"]].head()

# with open(os.path.join(root, 'questions', f'{dataset_type}_{split}_questions.json')) as f:
#         data = json.load(f)

# def read_json(path: pathlib.Path) -> dict:
#     with open(str(path), "r") as fp:  # fp is the file pointer
#         file = json.loads(s=fp.read())

#     return file


# y = read_json(target_ID)


# The below code does not work well, return None. 
# def get_filenames_of_path(path: pathlib.Path, ext: str = "*") -> List[pathlib.Path]:
#     """
#     Returns a list of files in a directory/path. Uses pathlib.
#     """
#     filenames = [file for file in path.glob(ext) if file.is_file()]
#     assert len(filenames) > 0, f"No files found in path: {path}"
#     return filenames

# root = pathlib.Path('/home/jingying/AIPython/data/headdata')
# targets = get_filenames_of_path(root / 'target')
# targets = targets.sort()
# print(targets)




# # defining the files directory and testing directory
# files_dir = '/home/jingying/AIPython/data/headdata/input'

# class FruitImagesDataset(torch.utils.data.Dataset):

#     def __init__(self, files_dir, width, height, transforms=None):
#         self.transforms = transforms
#         self.files_dir = files_dir
#         self.height = height
#         self.width = width
        
#         # sorting the images for consistency
#         # To get images, the extension of the filename is checked to be jpg
#         self.imgs = [image for image in sorted(os.listdir(files_dir))
#                         if image[-4:]=='.jpg']
        
        
#         # classes: 0 index is reserved for background
#         # self.classes = [_, 'apple','banana','orange']

#     def __getitem__(self, idx):
        
#         img_name = self.imgs[idx]
#         image_path = os.path.join(self.files_dir, img_name)
#         return image_path
#         print(image_path)
    
# dataset = FruitImagesDataset(files_dir, 224, 224)
# # print(image_path)

# root = Path("/home/jingying/AIPython/data/headdata/input")
# inputs = [file for file in root.iterdir() if file.is_file()]
# img = Image.open(inputs).convert("RGB")
# print(img)



# Split the dataset
# method 1: 
# training validation test split
inputs_train, inputs_valid, inputs_test = inputs[:12], inputs[12:16], inputs[16:]
targets_train, targets_valid, targets_test = targets[:12], targets[12:16], targets[16:]


# method 2: 
 # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# method3: 
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# train test split
test_split = 0.2
tsize = int(len(dataset)*test_split)
dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])




# # For Training
# images,targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images,targets)   # Returns losses and detections
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)  

