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
        # print(TEXT.vocab.stoi[t])
        vec.append(TEXT.vocab.stoi[t])
    # qvector[e.question] = vec
    print(vec)

print(qvector)

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

