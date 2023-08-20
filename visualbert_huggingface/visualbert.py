import torch, torchvision
import os
import sys
py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.environ['PATH'] += py_dll_path
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
import json
import cv2
from copy import deepcopy
from torchsummary import summary

from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers # jy changed from FastRCNNOutputs to FastRCNNOutputLayers
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg

import numpy as np
import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import init
from tqdm import tqdm
from pprint import pprint
import csv
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None)
import cv2
# pd.set_option('colheader_justify', 'left')

torch.cuda.is_available()

CUDA_VISIBLE_DEVICES = 1
CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

file_path ="/home/jingying/baseline/lora/qa/questions_logic2.json"
q = pd.read_json(file_path)
q.head()

