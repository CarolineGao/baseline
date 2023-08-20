from dataset import LoRADataset
import model
from model import get_object_detection_model
# torchvision libraries
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# not offcial library
from engine import train_one_epoch, evaluate
import utils



# Set cuda
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Setting the dataset and dataloader

# Load Dataset
root = '/home/jingying/AIPython/data/headdata'

# mapping: As our labels are strings, e.g. ‘head’, we should integer encode them accordingly.
mapping = {
    'head': 1,
}

# Transformations
transforms = torchvision.transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ]
)
"""transforms need to be changed, which convert both images and targets. Also need to change mean and stds."""

dataset = LoRADataset(root=root, transform=transforms, mapping=mapping)
# dataset_test = LoRADataset(root=root, transform=transforms, mapping=mapping)

print(len(dataset))


# Split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# train test split
test_split = 0.2
tsize = int(len(dataset)*test_split)
train_set = torch.utils.data.Subset(dataset, indices[:-tsize])
val_set = torch.utils.data.Subset(dataset, indices[-tsize:])

# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(
    dataset = train_set, 
    batch_size=2, 
    shuffle=True, 
    num_workers=2,
    collate_fn=utils.collate_fn)

val_loader = torch.utils.data.DataLoader(
    dataset = val_set, 
    batch_size=2, 
    shuffle=False, 
    num_workers=2,
    collate_fn=utils.collate_fn)

print(len(train_loader))

"""Why it is 8, split 20 dataset to 16, 4, why train_set is 8, and test_set is 2 rather than 16 and 4??"""
# Dataloader method2 - code good. but it is 8 rather than 16. 
# train_set, val_set = torch.utils.data.random_split(dataset, [16, 4])

# train_loader = DataLoader(
#     dataset = train_set, 
#     shuffle=shuffle, 
#     batch_size=batch_size, 
#     num_workers=num_workers, 
#     pin_memory=pin_memory)


# Training
num_classes = 2

# get the model using our helper function
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=1,
                                               gamma=0.1)

# training for 10 epochs
num_epochs = 2

for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=2)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, val_loader, device=device)

  