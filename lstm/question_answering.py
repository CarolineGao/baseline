# Fine tune QA with custom dataset - LoRA

# Preprocess data

def read_QA(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    questions = []
    answers = []
    # iterate through all data in squad data
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in qa['answers']:
                    # append data to lists
                    questions.append(question)
                    answers.append(answer)
    # return formatted data lists
    return questions, answers


[{'qas':

 [{'question': 'In what country is Normandy located?', 'id': '56ddde6b9a695914005b9628', 
'answers': [{'text': 'France', 'answer_start': 159}, 
{'text': 'France', 'answer_start': 159}, 
{'text': 'France', 'answer_start': 159},
 {'text': 'France', 'answer_start': 159}], 
 'is_impossible': False}, 


 {'question': 'When were the Normans in Normandy?', 'id': '56ddde6b9a695914005b9629',
'answers': [{'text': '10th and 11th centuries', 'answer_start': 94}, 
{'text': 'in the 10th and 11th centuries', 'answer_start': 87}, 
{'text': '10th and 11th centuries', 'answer_start': 94}, 
{'text': '10th and 11th centuries', 'answer_start': 94}], '
is_impossible': False}, 


{'question': 'From which countries did the Norse originate?', 'id': '56ddde6b9a695914005b962a', 
'answers': [{'text': 'Denmark, Iceland and Norway', 'answer_start': 256}, 
{'text': 'Denmark, Iceland and Norway', 'answer_start': 256}, 
{'text': 'Denmark, Iceland and Norway', 'answer_start': 256}, 
{'text': 'Denmark, Iceland and Norway', 'answer_start': 256}], 
'is_impossible': False},



 {'question': 'Who was the Norse leader?', 'id': '56ddde6b9a695914005b962b', 
 'answers': [{'text': 'Rollo', 'answer_start': 308}, {'text': 'Rollo', 'answer_start': 308}, 
 {'text': 'Rollo', 'answer_start': 308}, {'text': 'Rollo', 'answer_start': 308}], 'is_impossible': False},
{'question': 'What century did the Normans first gain their separate identity?', 'id': '56ddde6b9a695914005b962c', 
'answers': [{'text': '10th century', 'answer_start': 671},
 {'text': 'the first half of the 10th century', 'answer_start': 649}, {'text': '10th', 'answer_start': 671}, 
 

# Example
Image: 3 x 250 x 250   timestep: 1
Question: What food in the image taste sweet and original from New Zealand  12 words
Answer: pumpkin, kiwifruit     2 words 

batch_size = 3 

CNN: channels x height x weight
LSTM: samples, time_steps, features

