import h5py
import torch
from torchvision.models.resnet import ResNet, resnet101
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transforms import Scale
import sys
import os
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    return x

transform = transforms.Compose([
    Scale([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

class LORA(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        # self.length = len(os.listdir(os.path.join(root, 'images', split)))
        # 图片文件夹下有多余一个文件，.ipynb_checkpoint, 因此读取长度约束时，是101个但是只有100张图片，因此要filter只有名字是fruit的文件的长度. - yuanjing
        image_file_path = os.listdir(os.path.join(root, 'images', split))
        lennew = list(filter(lambda x: True if 'fruit' in x else False, image_file_path))  # lennew = filter(name = fruit), lennew是我自己命名的变量名, is not a method. 
        self.length = len(lennew)
        # print('print lenght', self.length)
        # print('.chekcpoint', os.listdir(os.path.join(root, 'images', split)))


    def __getitem__(self, index):
        if self.split == 'train':
            index = index + 0
        elif self.split == 'val':
            index = index + 100
        elif self.split == 'test':
            index = index + 141
        img = os.path.join(self.root, 'images',
                        self.split,
                        'fruit_{}.png'.format(index))
        img = Image.open(img).convert('RGB')
        return transform(img)

    def __len__(self):
        return self.length

batch_size = 20

resnet = resnet101(True).to(device)
resnet.eval()
resnet.forward = forward.__get__(resnet, ResNet)

def create_dataset(split):
    dataloader = DataLoader(LORA('data/lora', split), batch_size=batch_size,
                            num_workers=4)

    size = len(dataloader)

    print(split, 'total', size * batch_size)

    f = h5py.File('data/lora_{}_features.hdf5'.format(split), 'w', libver='latest')
    dset = f.create_dataset('data', (size * batch_size, 1024, 14, 14),
                            dtype='f4')

    with torch.no_grad():
        for i, image in tqdm(enumerate(dataloader)):
            image = image.to(device)
            features = resnet(image).detach().cpu().numpy()  # mac-net features
            # print(features.shape)  # gjy
            dset[i * batch_size:(i + 1) * batch_size] = features

    f.close()


create_dataset('train')
create_dataset('val')
create_dataset('test')
print('ok')