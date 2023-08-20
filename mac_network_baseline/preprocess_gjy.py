from dataset import GQA
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


def process_question(root, split, word_dic=None, answer_dic=None, dataset_type='lora'):
    
    file_path ="/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/lora.csv"
    header_list = ["image_index", "question_token", "answer"]

    lora = pd.read_csv(file_path, names = header_list)
    lora['image_index'] = lora.image_index.astype(str)
    
    words = lora.question_token.str.split(',')
    lora['question_token'] = words
    answers = lora['answer']
    lora = lora.assign(question_token=lora.question_token.apply(lambda x: list(map(int, x)) ))
    
    print(lora)

    result_1 = lora.values.tolist()
#     result = []
#     result.append(lora.series)

    print(result_1)
    
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    # with open(os.path.join(root, 'questions', f'{dataset_type}_{split}_questions.json')) as f:
    #     data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    # for question in tqdm.tqdm(lora['questions']):
    #     words = nltk.word_tokenize(question['question'])

    question_token = []
    for word in words:
        try:
            question_token.append(word_dic[word])

        except:
            question_token.append(word_index)
            word_dic[word] = word_index
            word_index += 1

    answer_word = answers
    try:
        answer = answer_dic[answer_word]
    except:
        answer = answer_index
        answer_dic[answer_word] = answer_index
        answer_index += 1
        
    # result.append((question[image_index[dataset_type]], question_token, answer)) # Map LoRA with line 52. 
    # type(result)

    with open(f'data/{dataset_type}_{split}_new.pkl', 'wb') as f:
        pickle.dump(result_1, f)

    return word_dic, answer_dic


if __name__ == '__main__':
    dataset_type = 'gqa'
    root = 'gqa'
    word_dic, answer_dic = process_question(root, 'train', dataset_type=dataset_type)
    process_question(root, 'val', word_dic, answer_dic, dataset_type=dataset_type)

    with open(f'data/{dataset_type}_dic_new.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
        
        
        
      
