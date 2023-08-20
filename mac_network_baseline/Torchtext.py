import spacy
import numpy as np
import pandas as pd
from numpy import array
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
from spacy.lang.en import English
import en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")
import torch
import torch.nn as nn
from torchtext import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
import os
from pprint import pprint
import sys
import numpy
pd.set_option('display.max_rows', None, 'display.max_columns', None)

# Read csv file， 读问题
file_path ="/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/question_token.csv"
# file_write_path = "/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/question_token_write.csv"
file_csv = pd.read_csv(file_path)
lora = pd.DataFrame(file_csv, columns=["question", "answer"])
lora

# Split to train and test dataset
train, test = train_test_split(lora, test_size=0.1)
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
pprint(train.to_csv)
pprint(test.to_csv)

# 提取字段
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# Field定义如何处理字段
TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True)

# fields把csv中的question列和TEXT如何处理字段匹配上
fields = [("question", TEXT)] # "question" is the field of column head in csv file. TEXT is the Field name. 
# print(fields)

# 用TabularDataset来匹配下train和test数据所对应的fields
train, test = TabularDataset.splits(
    path="", train="train.csv", test="test.csv", format="csv", skip_header = True, fields=fields)

# 创建vocab
TEXT.build_vocab(train, vectors="glove.6B.100d")  # TEXT is the vairable name of Field. 

# Print （创建vocab后，就可以打印字段的index数字）
print(train[4].__dict__, '\n')
# print(train[0].__dict__.keys(), '\n')
# print(train[0].question)
# print(train[0].__dict__.values())

# # vocab.stoi是字对应数字，vocab.itos是数字对应字
# print(TEXT.vocab.stoi['which'])
# print(TEXT.vocab.itos[7])
# print(len(TEXT.vocab.itos))
# print(TEXT.vocab.itos[:50])
# print(TEXT.vocab.stoi)

qvector = {}
for e in train.examples:
    vec = []
    for t in e.question:
        print(TEXT.vocab.stoi[t])
        vec.append(TEXT.vocab.stoi[t])
    qvector[e.question] = vec

print(qvector)

# 每个token和index
for token, index in TEXT.vocab.stoi.items():
    print(token, index)

# TO DO 打印csv文件中每条问题所对应的index， 例如[7， 4， 3， 2， 1， 9]，保存到list中
# 以下为示例代码网上找到的（作为参考，不能完善跑通）：

# saving the vocabulary used for training to a text file (tab separated):
def save_vocab(vocab, file_path):
    with open(file_path, 'w+') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}')
            
            
# in the inference file, we can the vocabulary into a plain old Python dictionary:
def read_vocab(path):
    vocab = dict()
    with open(path, 'r') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab


# 
def predict_sentiment(model, sentence, vocab, unk_idx):
    """
    model is your PyTorch model
    sentence is string you wish to predict sentiment on
    vocab is dictionary, keys = tokens, values = index of token
    unk_idx is the index of the <unk> token in the vocab
    """
    tokens = tokenize(sentence) #convert string to tokens, needs to be same tokenization as training
    indexes = [vocab.get(t, unk_idx) for t in tokens] #converts to index or unk if not in vocab
    tensor = torch.LongTensor(indexes).unsqueeze(1) #convert to tensor and add batch dimension
    output = model(tensor) #get output from model
    prediction = torch.sigmoid(output) #squeeze between 0-1 range
    return prediction


# # 以下为mac-network原有的提取token和index，并且保存的代码，作为参考
# with open(os.path.join(root, 'questions', f'{dataset_type}_{split}_questions.json')) as f:
#         data = json.load(f)

#     result = []
#     word_index = 1
#     answer_index = 0

#     for question in tqdm.tqdm(data['questions']):
#         words = nltk.word_tokenize(question['question'])
#         question_token = []

#         for word in words:
#             try:
#                 question_token.append(word_dic[word])

#             except:
#                 question_token.append(word_index)
#                 word_dic[word] = word_index
#                 word_index += 1

#         answer_word = question['answer']

#         try:
#             answer = answer_dic[answer_word]
#         except:
#             answer = answer_index
#             answer_dic[answer_word] = answer_index
#             answer_index += 1
        
#         result.append((question[image_index[dataset_type]], question_token, answer))



# # 以下为把数据集导入model中，训练文本quesiton数据集，获得vector
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_iter, valid_iter, test_iter = BucketIterator.splits(
#     (train, valid, test),
#     batch_sizes=(64, 64, 64),
#     device=device,
#     sort_key=lambda x: len(x.text),
#     sort_within_batch=True
# )


train_iterator, test_iterator = BucketIterator.splits(
    (train, test), batch_size=3, device="cuda"
)

for batch in train_iterator:
    print("tensor", batch.question)

    
# print("vocab", question.vocab.stoi)

