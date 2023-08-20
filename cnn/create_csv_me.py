import pandas as pd
import os

path = "/home/jingying/AIPython/data/lora/"
train_df = pd.DataFrame(columns=['img_name','label'])
train_df['img_name'] = os.listdir(path + "train/")
# print(train_df)

for idx, i in enumerate(os.listdir(path + "train")): 
    print(idx)
    if "cat" in i:
        train_df["label"][idx] = 0
    if "dog" in i:
        train_df["label"][idx] = 1
    
print(train_df)

train_df.to_csv(r'cnn/train_me_csv.csv', index = False, header=True)



# train_df.to_csv(r'cnn/train_me1_csv.csv', index = False, header=True)