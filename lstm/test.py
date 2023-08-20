import torch
import numpy as np

x = np.random.randn(2, 3)
print(x.max(1))
print(torch.from_numpy(x).max(1)[0])