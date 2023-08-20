# Custom Datasets for Text
# Dataset we are using to demo: Flickr
# /home/jingying/AIPython/data/Flickr8k

# Convert text -> numerical values 
# 1. Need a Vocabulary mapping each word to an index
# 2. Setup pytorch dataset to load the data. 
# 3. Setup padding, keep same length

# ToDo -WIP NOT Finished

# Import
import os
import pandas as pd
import spacy  # for tokenizer 
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img

spacy_eng = spacy.load("en_core_web_sm")

# class TextDataset(Dataset):
#     def __init__(self):
#         self.root_dir = root_dir
#         self.df 
#         self.transform = transform


#     def __getitem__(self, index: Any):
#         return super().__getitem__(index)

#     def __len__(self):
#         return len()


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


path = "/home/jingying/AIPython/data/Flickr8k/captions.txt"
df = pd.read_csv(path)
# print(df)
captions = df["caption"]
# print(captions)
caption  =captions[4]
print(caption)


# dfi = df["image"]
# # print(dfi)
# print(dfi[1])
# img = Image.open("/home/jingying/AIPython/data/Flickr8k/images/1000268201_693b08cb0e.jpg").convert("RGB")
# print(img)


# convert to numerical 
vocab = Vocabulary(1)
# length = len(vocab) # output: 4
# print(length) # output: 4
# print(vocab)
# 

# print(len(captions))

list = captions.tolist()
print(len(list))
# print(list)

# a = vocab.build_vocabulary(captions.tolist())
# print(a)

# caption.stoi()

numericalized_caption = [vocab.stoi["<SOS>"]]
numericalized_caption += vocab.numericalize(caption)
numericalized_caption.append(vocab.stoi["<EOS>"])

print(numericalized_caption)