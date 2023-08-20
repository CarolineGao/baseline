#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
torch.__version__


# In[2]:


torch.cuda.is_available()


# In[3]:


CUDA_VISIBLE_DEVICES = 1
CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[4]:


# %%capture
# # !pip install pyyaml==5.1
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# # See https://detectron2.readthedocs.io/tutorials/install.html for instructions


# ### Imports

# In[5]:


import os
import sys
py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.environ['PATH'] += py_dll_path


# In[6]:


import torch, torchvision
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from copy import deepcopy
from torchsummary import summary


# In[7]:


from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments


# In[8]:


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


# In[9]:


import numpy as np
import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import init
from tqdm import tqdm
import os
from pprint import pprint
import sys
import json
import csv
import sys
import json
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None)
from matplotlib import pyplot as plt
import cv2
# pd.set_option('colheader_justify', 'left')


# ### Load Examples
# The next few cells show how to get an example from the VQA v2 dataset. We will only use the image from the example.

# In[10]:


# with open('/home/jingying/AIPython/data/VQA/v2_OpenEnded_mscoco_val2014_questions.json') as f:
#     q = json.load(f)


# In[11]:


# with open('/home/jingying/AIPython/data/VQA/questions.json') as f:
#     q = json.load(f)


# In[12]:


file_path ="/home/jingying/baseline/lora/qa/questions_logic2.json"
q = pd.read_json(file_path)
# q.head()
print(q)


# In[13]:


idx = 1


# In[14]:


question1 = q["question"][idx]
question1


# In[15]:


file_path ="/home/jingying/baseline/lora/qa/annotations_logic2.json"
a = pd.read_json(file_path)
a.head()


# In[16]:


answer_word1 = a["answers"][idx]
answer_word1


# In[17]:


image_id = q['image_id'][idx]
image_id


# In[18]:


idx = 1
question1 = q["question"][idx]
answer_word1 = a["answers"][idx]
image_id = q['image_id'][idx]


# In[19]:


img1 = plt.imread(f'/home/jingying/baseline/lora/data/lora_train_1000.png')
# Detectron expects BGR images
img_bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)


# In[20]:


img_bgr1


# In[21]:


print(img1.shape)


# In[22]:


plt.imshow(img1)
plt.show()


# In[23]:


question1


# In[24]:


answer_word1


# ### Taking another image for a "batch"

# In[25]:


idx = 2

question2 = q["question"][idx]
answer_word2 = a["answers"][idx]
image_id = q['image_id'][idx]

img2 = plt.imread(f'/home/jingying/baseline/lora/data/lora_train_1001.png')
# Detectron expects BGR images
img_bgr2 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

# Detectron expects BGR images
plt.imshow(img2)
plt.show()


# In[26]:


img2.shape # Note that images are differently-sized


# In[27]:


image_id


# In[28]:


question2


# In[29]:


answer_word2


# ### Load Config and Model Weights

# I am using the MaskRCNN ResNet-101 FPN checkpoint, but you can use any checkpoint of your preference. This checkpoint is pre-trained on the COCO dataset. You can check other checkpoints/configs on the [Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) page.

# In[30]:


cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
from detectron2.config import get_cfg

def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
#     cfg['MODEL']['DEVICE']='cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

    return cfg

cfg = load_config_and_model_weights(cfg_path)


# ### Load the Object Detection Model
# The `build_model` method can be used to load a model from the configuration, the checkpoints have to be loaded using the `DetetionCheckpointer`.

# In[31]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[32]:


def get_model(cfg):
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.to(device)
    return model

model = get_model(cfg)
# CC no training parameters. 
for param in model.parameters():
    param.requires_grad = False


# ### Convert Image to Model Input
# The detectron uses resizing and normalization based on the configuration parameters and the input is to be provided using `ImageList`. The `model.backbone.size_divisibility` handles the sizes (padding) such that the FPN lateral and output convolutional features have same dimensions.

# In[33]:


# Test code
img1 = plt.imread(f'/home/jingying/git_paper_1/image_generation/output_baseline/lora_train_1000.png')
img_bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
img_list = [img_bgr1, img_bgr2]
img_list
# images, batched_inputs = prepare_image_inputs(cfg, [img_bgr1, img_bgr2])


# In[34]:


# Batch deal with images - works code
directory = '/home/jingying/baseline/lora/data'

img_list = []
for img in os.listdir(directory):
    if img.endswith(".png"):
        img = plt.imread(f'/home/jingying/baseline/lora/data/{img}')
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_list.append(img_bgr)
print(img_list)


# In[35]:


def prepare_image_inputs(cfg, img_list):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1)).cuda()

    batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1).cuda()
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1).cuda()
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]

    # Convert to ImageList
    images =  ImageList.from_tensors(images,model.backbone.size_divisibility)
#     images = images.cuda() # add jingying
    
    return images, batched_inputs

# images, batched_inputs = prepare_image_inputs(cfg, [img_bgr1, img_bgr2])
images, batched_inputs = prepare_image_inputs(cfg, img_list)


# In[36]:


print(len(batched_inputs))


# ### Get ResNet+FPN features
# The ResNet model in combination with FPN generates five features for an image at different levels of complexity. For more details, refer to the FPN paper or this [article](https://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd). For this tutorial, just know that `p2`, `p3`, `p4`, `p5`, `p6` are the features needed by the RPN (Region Proposal Network). The proposals in combination with `p2`, `p3`, `p4`, `p5` are then used by the ROI (Region of Interest) heads to generate box predictions.

# In[37]:


def get_features(model, images):
    features = model.backbone(images.tensor)
    return features
features = get_features(model, images)


# In[38]:


features.keys()


# ### Visualizing Image and Image features
# Just for a sanity check, we visualize the 0th channels in each of the features, and their shapes.

# In[39]:


plt.imshow(cv2.resize(img2, (images.tensor.shape[-2:][::-1])))
plt.show()
for key in features.keys():
    print(features[key].shape)
    plt.imshow(features[key][1,0,:,:].squeeze().detach().cpu().numpy(), cmap='jet')
    plt.show()


# ### Get region proposals from RPN
# This RPN takes in the features and images and generates the proposals. Based on the configuration we chose, we get 1000 proposals.

# In[40]:


def get_proposals(model, images, features):
    proposals, _ = model.proposal_generator(images, features)
    return proposals

proposals = get_proposals(model, images, features)


# ### Get Box Features for the proposals
# 
# The proposals and features are then used by the ROI heads to get the predictions. In this case, the partial execution of layers becomes significant. We want the `box_features` to be the `fc2` outputs of the regions. Hence, I use only the layers that are needed until that step. 

# In[41]:


def get_box_features(model, features, proposals):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)

    box_features = box_features.reshape(len(batched_inputs), 1000, 1024) # depends on your config and batch size
    return box_features, features_list

box_features, features_list = get_box_features(model, features, proposals)


# ### Get prediction logits and boxes
# The prediction class logits and the box predictions from the ROI heads, this is used in the next step to get the boxes and scores from the `FastRCNNOutputs`
# 

# In[42]:


def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas

pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)


# ### Get FastRCNN scores and boxes
# 
# This results in the softmax scores and the boxes.

# In[43]:


pred_class_logits.shape


# In[44]:


len(proposals)


# In[45]:


def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas):
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    outputs = FastRCNNOutputLayers(
        input_shape = 2,
        box2box_transform=box2box_transform,
        num_classes = pred_class_logits.shape[1],
#         pred_class_logits = pred_class_logits,
#         pred_proposal_deltas = pred_proposal_deltas,
#         proposals = proposals,
        smooth_l1_beta = smooth_l1_beta,
    )

    boxes = outputs.predict_boxes((pred_class_logits, pred_proposal_deltas), proposals)
    scores = outputs.predict_probs((pred_class_logits, pred_proposal_deltas), proposals)
    image_shapes = [x.image_size for x in proposals]

    return boxes, scores, image_shapes

boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas)


# In[46]:


boxes


# ### Rescale the boxes to original image size
# We want to rescale the boxes to original size as this is done in the detectron2 library. This is done for sanity and to keep it similar to the visualbert repository.

# In[47]:


def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes

output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]


# ### Select the Boxes using NMS
# We need two thresholds - NMS threshold for the NMS box section, and score threshold for the score based section.
# 
# First NMS is performed for all the classes and the max scores of each proposal box and each class is updated.
# 
# Then the class score threshold is used to select the boxes from those.

# In[48]:


def select_boxes(cfg, output_boxes, scores):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    cls_boxes = output_boxes.tensor.detach().reshape(1000,80,4)
    max_conf = torch.zeros((cls_boxes.shape[0])).to(scores)
    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        det_boxes = cls_boxes[:,cls_ind,:]
        keep = nms(det_boxes, cls_scores, test_nms_thresh)
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf


# In[49]:


temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
keep_boxes, max_conf = [],[]
for keep_box, mx_conf in temp:
    keep_boxes.append(keep_box)
    max_conf.append(mx_conf)


# ### Limit the total number of boxes
# In order to get the box features for the best few proposals and limit the sequence length, we set minimum and maximum boxes and pick those box features.

# In[50]:


MIN_BOXES=10
MAX_BOXES=100
def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    max_conf = max_conf.cpu()
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
    return keep_boxes

keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]


# ### Get the visual embeddings :) 
# Finally, the boxes are chosen using the `keep_boxes` indices and from the `box_features` tensor.

# In[51]:


def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]

visual_embeds = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]


# ## Tips for putting it all together
# 
# Note that these methods can be combined into different parts to make it more efficient: 
# 1. Get the model and store it in a variable.
# 2. Transform and create batched inputs separately.
# 3. Generate visual embeddings from the detectron on the batched inputs and models.
# 
# Ideally, you want to build a class around this for ease of use - The class should contain all the methods, the model and the configuration details. And it should process a batch of images and convert to embeddings.

# ## Using the embeddings with VisualBert

# In[52]:


from getpass import getpass
import urllib
# %cd /content/
# user = input('User name: ')
# password = getpass('Password: ')
# password = urllib.parse.quote(password) # your password is converted into url format
# cmd_string = f'git clone -b add_visualbert --single-branch https://{user}:{password}@github.com/gchhablani/transformers.git'
# os.system(cmd_string)
# cmd_string, password = "", "" # removing the password from the variable
# %cd transformers
# !pip install -e ".[dev]"
# !pip install transformers


# In[53]:


from transformers import BertTokenizer, VisualBertForPreTraining


# In[54]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[55]:


# Batch deal with questions
file_path ="/home/jingying/git_paper_1/vqa_integration/baseline_vqa_virtualbert/questions_logic2.json"
q = pd.read_json(file_path)
q.head()

questions = []
for idx in range(len(q)):
    question = q["question"][idx]
    questions.append(question)

# print(type(questions))


# In[56]:


# questions = [question1, question2]
tokens = tokenizer(questions, padding='max_length', max_length=150)


# In[57]:


input_ids = torch.tensor(tokens["input_ids"]).cuda()
attention_mask = torch.tensor(tokens["attention_mask"]).cuda()
token_type_ids = torch.tensor(tokens["token_type_ids"]).cuda()


# In[58]:


# Batch deal with answers
file_path ="/home/jingying/git_paper_1/vqa_integration/baseline_vqa_virtualbert/annotations_logic2.json"
a = pd.read_json(file_path)
a.head()


answers = []
for idx in range(len(a)):
    answer_word = a["answers"][idx]
    answers.append(answer_word)

print(answers)


# In[59]:


# answers = [answer_word1, answer_word2]
answer_tokens = tokenizer(answers, padding='max_length', max_length=150)
answer_token_type_ids = torch.tensor(answer_tokens["token_type_ids"]).cuda()


# In[60]:


visual_embeds = torch.stack(visual_embeds).cuda()
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=torch.device("cuda"))
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long,device=torch.device("cuda"))


# In[61]:


def print_network(net):
    num_params = 0
    trainable = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total num of parameters: %d' % num_params)
    for param in net.parameters():
        if param.requires_grad:
            trainable += param.numel()
    print('Total num of trainable parameters: %d' % trainable)
print_network(model)


# In[62]:


model_final= VisualBertForPreTraining.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre').cuda() # this checkpoint has 1024 dimensional visual embeddings projection


# In[63]:


# for name, param in model_final.named_parameters():
#     if name == 'encoder.embedding.weight':
#         param.requires_grad = False

pre_dict = model_final.state_dict()
# unfreeze_list = []
for k,param in pre_dict.items():
    print(k)
    print(param.shape)
    if  "seq_relationship" not in k:
        param.requires_grad = False


# In[64]:


# a = torch.Tensor(150).long()
# summary(model_final, input_size=(150,), batch_size=1)


# In[65]:


outputs = model_final(input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
                visual_embeds=visual_embeds, 
                visual_attention_mask=visual_attention_mask, 
                visual_token_type_ids=visual_token_type_ids)


# In[66]:


def print_network(net):
    num_params = 0
    trainable = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total num of parameters: %d' % num_params)
    for param in net.parameters():
        if param.requires_grad:
            trainable += param.numel()
    print('Total num of trainable parameters: %d' % trainable)
print_network(model_final)


# In[67]:


import torch
torch.cuda.empty_cache()


# In[68]:


input_ids.shape


# In[69]:


# outputs


# In[70]:


attention_mask.shape, token_type_ids.shape, visual_embeds.shape, 
visual_attention_mask.shape, visual_token_type_ids.shape,input_ids.shape


# In[71]:


from torch.utils.data import Dataset, DataLoader


# In[72]:


a = torch.cat([answer_token_type_ids[0],visual_attention_mask[0]])


# In[73]:


class FeatureDataset(Dataset):
    def __init__(self,input_ids, 
                 attention_mask,
                 token_type_ids,
                 visual_embeds,
                 visual_attention_mask,
                 visual_token_type_ids,
                 answer_token_type_ids):
        super().__init__()
        self.input_ids= input_ids.detach().cpu()
        self.attention_mask = attention_mask.detach().cpu()
        self.token_type_ids = token_type_ids.detach().cpu()
        self.visual_embeds = visual_embeds.detach().cpu()
        self.visual_attention_mask = visual_attention_mask.detach().cpu()
        self.visual_token_type_ids = visual_token_type_ids.detach().cpu()
        self.answer_token_type_ids = answer_token_type_ids.detach().cpu()
        
    def __getitem__(self,i):
        """
        id.pkl 
        """
        return {
            "input_ids":self.input_ids[i], 
            "attention_mask": self.attention_mask[i],
            "token_type_ids": self.token_type_ids[i],
            "visual_embeds": self.visual_embeds[i],
            "visual_attention_mask": self.visual_attention_mask[i],
            "visual_token_type_ids": self.visual_token_type_ids[i],
            "labels":torch.cat([self.answer_token_type_ids[0],self.visual_attention_mask[0]])}
    def __len__(self):
        return len(self.attention_mask)


# In[74]:


# answer_token_type_ids[0].shape,visual_attention_mask[0].shape


# In[75]:


# a.shape


# In[76]:


# outputs.items()


# In[77]:


# outputs


# In[78]:


processed_train_data = FeatureDataset(input_ids,
                 attention_mask,
                 token_type_ids,
                 visual_embeds,
                 visual_attention_mask,
                 visual_token_type_ids,
                 answer_token_type_ids)
processed_eval_data = FeatureDataset(input_ids,
                 attention_mask,
                 token_type_ids,
                 visual_embeds,
                 visual_attention_mask,
                 visual_token_type_ids,
                 answer_token_type_ids)
processed_test_data = FeatureDataset(input_ids,
                 attention_mask,
                 token_type_ids,
                 visual_embeds,
                 visual_attention_mask,
                 visual_token_type_ids,
                 answer_token_type_ids)


# In[79]:


for x in processed_train_data:
    for key,value in x.items():
        print(value.shape)
    break


# In[80]:


answer_token_type_ids.shape


# In[81]:


len(answer_tokens['input_ids'][1])


# In[82]:


outputs['prediction_logits'].shape


# In[83]:


# for e in range(epochs):
#     for images, labels in train_loader:   
#         if torch.cuda.is_available():
#             images, labels = images.cuda(), labels.cuda()   
#         # blablabla  


# In[84]:


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='model_results',          # output directory
    overwrite_output_dir=True,
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=20,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=None,            # directory for storing logs
    logging_steps=10
)

trainer = Trainer(
    model=model_final, # the instantiated Transformers model to be trained
    args=training_args, # training arguments, defined above
    train_dataset=processed_train_data, # training dataset
    eval_dataset=processed_test_data, # evaluation dataset
    compute_metrics=lambda *args,**kwargs: 0,            
)

trainer.train()


# In[85]:


trainer.eval()


# In[ ]:




