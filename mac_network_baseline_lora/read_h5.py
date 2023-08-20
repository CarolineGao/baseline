import h5py
filename = "/home/jingying/baseline/mac-network-pytorch/mac-network-pytorch-gqa/data/lora_train_features.hdf5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    print(f["data"])
    print(len(f["data"]))
    # print(type(f["data"]))
   