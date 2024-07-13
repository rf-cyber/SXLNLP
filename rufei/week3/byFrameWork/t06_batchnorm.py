import torch

import numpy as np

x = np.random.random((4, 5))

bn = torch.nn.BatchNorm1d(5)

y = bn(torch.from_numpy(x).float())

print(x)
print("torch y:", y)
print(bn.state_dict())
