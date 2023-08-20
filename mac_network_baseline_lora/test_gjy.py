# test_gjy.py is for Testing the model input and output. 

from model_gqa import MACNetwork
import sys
import pickle
from collections import Counter
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CLEVR, LORA, collate_data, transform
from dataset import GQA, collate_data, transform  # JY added -26_Jul_2021
from model_gqa import MACNetwork

# from train.py import net, net_running -gjy
# print(sys.argv[0],sys.argv[1]) -gjy

batch_size = 64
n_epoch = 180


dim_dict = {'CLEVR': 512,
            'gqa': 2048,
            'lora': 2048}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

datapath = '/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/'
modelpath = '/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/checkpoint/checkpoint_lora25.model'

dataset_type = 'lora'
with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
    dic = pickle.load(f)

n_words = len(dic['word_dic']) + 1
n_answers = len(dic['answer_dic'])

net = MACNetwork(n_words, dim_dict[dataset_type], classes=n_answers, max_step=4).to(device)
net_running = MACNetwork(n_words, dim_dict[dataset_type], classes=n_answers, max_step=4).to(device)

print(net_running.state_dict())

# The below is test model code. 
test_set = DataLoader(
    #CLEVR(sys.argv[1], 'val', transform=None),
    LORA(datapath, 'val', transform=None),
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_data,
)

net_running.load_state_dict(torch.load(modelpath))
# net = torch.load(sys.argv[2])
net_running.eval()  
net_running.to(device)

for epoch in range(n_epoch):
    dataset = iter(test_set)
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
