import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model_me import CNN
from dataset_me import CatDogDataset
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = ("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Hyperparameters
"""
pin_memory: 
Custom dataset was created on CPU and all operations/data running in CPU, when training, the batches of images will be moved to GPU. 
pin_memory ensures that this movement of data is efficient and fast. 

"""


num_epochs = 2
learning_rate = 0.00001
train_CNN = False
batch_size = 1
shuffle = True
pin_memory = True 
num_workers = 1


# Setting the dataset and dataloader
dataset = CatDogDataset(
    root_dir="/home/jingying/AIPython/data/lora/train",
    annotation_file="/home/jingying/baseline/cnn/train_me.csv",
    transform=transform,
)

print(len(dataset))
train_set, val_set = torch.utils.data.random_split(dataset, [8, 4])

train_loader = DataLoader(
    dataset = train_set, 
    shuffle=shuffle, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    pin_memory=pin_memory)

val_loader = DataLoader(
    dataset = val_set, 
    shuffle=shuffle, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    pin_memory=pin_memory)


model = CNN().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

for name, param in model.inception.named_parameters():
    if "fc.weight" in name or "fc.bias" in name:
        param.requires_grad = True
    else: 
        param.requires_grad = train_CNN

"""
The flag which we set earlier is now being used to set the fc layers to trainable 
and all other layers to non — trainable to avoid back-propagation through those layers. 
The CNN().to(device) moves the model to GPU. Note for GPU training both the model and data must be loaded to the GPU. 
Refer to torch docs for input formats for BCELoss and Adam optimizer.
"""


# Train the model
# total_step = len(train_loader)
# for epoch in range(epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Accuracy Check
def check_accuracy(loader, model):
    if loader == train_loader:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on validation data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
            # predictions = scores.argmax(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return f"{float(num_correct)/float(num_samples)*100:.2f}"
    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
    
    model.train()


def train():
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, total = len(train_loader), leave = True)
        if epoch % 2 == 0:
            loop.set_postfix(val_acc = check_accuracy(val_loader, model))

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())

if __name__ == "__main__":
    train()







# Test the model
# model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
