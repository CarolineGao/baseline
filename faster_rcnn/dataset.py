import os
import pathlib
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import json
from PIL import Image
from skimage import io
from skimage.color import rgba2rgb
from skimage.io import imread
import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from utils import read_json, map_class_to_int, get_transform
from typing import List, Dict
# from transformations import ComposeDouble, Clip, AlbumentationWrapper, FunctionWrapperDouble
# from transformations import normalize_01
import numpy as np
import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torchvision.transforms as transforms  



# from pytorch_faster_rcnn_tutorial.transformations import (
#     ComposeDouble,
#     ComposeSingle,
#     map_class_to_int,
# )
# from pytorch_faster_rcnn_tutorial.utils import 

"""

The dataset __getitem__ should return:

image: a PIL Image of size (H, W)
target: a dict containing the following fields
-boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
-labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
-image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
-area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
-iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
-(optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
-(optionally) keypoints (FloatTensor[N, K, 3]): For each one of the N objects, it contains the K keypoints in [x, y, visibility] format, defining the object. 

"""

class LoRADataset(Dataset):
    def __init__(self, root, transform=None, mapping:Dict = None):
        self.root = root
        self.imgs = [image for image in sorted(os.listdir(os.path.join(root, "input"))) if image[-4:]=='.jpg']
        self.annotations = [file for file in sorted(os.listdir(os.path.join(root, "target")))]
        self.transform = transform
        self.mapping = mapping


    def __len__(self):
        # return len(self.annotations)
        return len(self.imgs) 

    def __getitem__(self, index):

        # Get images info by iterating image and convert image to RGB formate
        img_name = self.imgs[index]
        img_path = os.path.join(self.root, "input", img_name)
        img = Image.open(img_path).convert("RGB")
        # image = io.imread(img_path)
        # image = Image.fromarray(image)

        # Get annotations info 
        # Select the sample
        target_ID = self.annotations[index]
        
        # Load target & Read json file
        target_path = os.path.join(self.root, "target", target_ID)
        targets = read_json(target_path)

        # Read boxes
        try:
            boxes = torch.from_numpy(targets["boxes"]).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(targets["boxes"]).to(torch.float32)
            
        # Read labels - Label Mapping from str to integer
        if self.mapping:
            labels = map_class_to_int(targets["labels"], mapping=self.mapping)
        else:
            labels = targets["labels"]

        # Read labels - Label convert to numpy
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)


        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([index])
        target["image_id"] = image_id
    
        # Preprocessing
        # target = {key: value.numpy() for key, value in target.items()}  # all tensors should be converted to np.ndarrays

        # if self.transform is not None:
        #     img, target = self.transform(img, target)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target


# # Load Dataset
# root = '/home/jingying/AIPython/data/headdata'

# # # mapping: As our labels are strings, e.g. ‘head’, we should integer encode them accordingly.
# mapping = {
#     'head': 1,
# }

# # transforms = ComposeDouble([
# #     Clip(),
# #     # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
# #     # AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
# #     # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
# #     FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
# #     FunctionWrapperDouble(normalize_01)
# # ])

# # transforms = get_transform(train=True)

# # Transformations
# transforms = torchvision.transforms.Compose(
#     [
#         transforms.Resize((256, 256)),
#         transforms.RandomCrop((250, 250)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
#     ]
# )
# """transforms need to be changed, which convert both images and targets. Also need to change mean and stds."""

# dataset = LoRADataset(root=root, transform=transforms, mapping=mapping)

# print(len(dataset))

# img, target = dataset[1]
# print(img.shape, '\n',target)

# def plot_img_bbox(img, target):
#     # plot the image and bboxes
#     # Bounding boxes are defined as follows: x-min y-min width height
#     fig, a = plt.subplots(1,1)
#     fig.set_size_inches(5,5)
#     a.imshow(img)
#     for box in (target['boxes']):
#         x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
#         rect = patches.Rectangle((x, y),
#                                  width, height,
#                                  linewidth = 2,
#                                  edgecolor = 'r',
#                                  facecolor = 'none')

#         # Draw the bounding box on top of the image
#         a.add_patch(rect)
#     plt.show()
    
# # plotting the image with bboxes. Feel free to change the index
# img, target = dataset[1]
# plot_img_bbox(img, target)