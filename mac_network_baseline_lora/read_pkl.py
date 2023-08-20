import pickle as pickle

fr = open('/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/gqa_val.pkl','rb')    #open的参数是pkl文件的路径
inf = pickle.load(fr)      #读取pkl文件的内容
fr.close()