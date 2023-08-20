import os
import sys
import json
import pickle
import nltk
import tqdm
from torchvision import transforms
from PIL import Image
from transforms import Scale
import pandas as pd


image_index = {'CLEVR': 'image_filename',
               'gqa': 'imageId',
               'lora': 'imageId'}


def process_question(root, split, word_dic=None, answer_dic=None, dataset_type='lora'):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    """
    Need to seperate question and answers first from csv file before feed into the pkl here. 
    Generate train.pkl and val.pkl seperately. 
    """

    # with open(os.path.join(root, 'questions', f'{dataset_type}_{split}_questions.json')) as f:
    #     data = json.load(f)

    file_path ="/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/lora_train.csv"
    header_list = ["question", "answer"]

    lora = pd.read_csv(file_path, names = header_list)
    
    result = []
    word_index = 1
    answer_index = 0
    nltk.download('punkt')
    for question in lora.iterrows():
        q = question[1]['question']
        if len(q) < 20:
            continue
        print('-----',question[1]['question'],'---------')
        words = nltk.word_tokenize(q)
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        a = question[1]['answer']
        answer_word = a

        try:
            answer = answer_dic[answer_word]
        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1
        
        result.append((question[0], question_token, answer)) # Map LoRA with line 52. 

    with open(f'data/{dataset_type}_{split}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic


if __name__ == '__main__':
    dataset_type = 'lora_test'
    root = 'data'
    word_dic, answer_dic = process_question(root, 'train', dataset_type=dataset_type)
    process_question(root, 'val', word_dic, answer_dic, dataset_type=dataset_type)

    with open(f'data/{dataset_type}_dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
