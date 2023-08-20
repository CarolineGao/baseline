import json
import csv
import pandas as pd
import numpy as np

# 读复杂层级的json文件

file_path ="/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/gqa/all_train_data.json"

# with open(file_path) as project_file:    
#     data = json.load(project_file)  

# df = pd.json_normalize(data)
# df.head()

data = json.load(open(file_path))
df = pd.DataFrame(data["questions"])
df.head()
