import sys
# print(sys.argv[0],sys.argv[1])   - JY
import pickle
from collections import Counter

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CLEVR, LORA, collate_data, transform
from dataset import GQA, collate_data, transform  # JY added -26_Jul_2021
from model_gqa import MACNetwork

batch_size = 64
n_epoch = 2
path = '/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/checkpoint/checkpoint_lora25.model'
datapath = '/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/'

test_set = DataLoader(
    #CLEVR(sys.argv[1], 'val', transform=None),
    LORA(datapath, 'val', transform=None),
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_data,
)

with open(f'data/lora_val.pkl', 'rb') as f:
    dic = pickle.load(f)

n_words = len(dic['word_dic']) + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MACNetwork(n_vocab=n_words,dim=2048).to(device)
net = torch.load(path) 
model.load_state_dict(net)
# net.eval()  

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(n_epoch):
    dataset = iter(train_set)
    pbar = tqdm(dataset)
    correct_counts = 0
    total_counts = 0

    for image, question, q_len, answer in pbar:
        image, question = image.to(device), question.to(device)
        output = net(image, question, q_len)
        correct = output.detach().argmax(1) == answer.to(device)
        for c in correct:
            if c:
                correct_counts += 1
            total_counts += 1

    print('Avg Acc: {:.5f}'.format(correct_counts / total_counts))
