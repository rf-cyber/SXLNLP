# coding:utf-8
import torch

x = [[1, 2, 3], [4, 5, 6]]

x2 = torch.Tensor(x)
x2.numpy()

print("x2： ", x2)
print("x2张量移动到GPU上： ", x2.cuda(0))
print("X2张量移从GPU动到CPU上： ", x2.cuda(0).cpu())
print("x2张量转numpy数组： ", x2.cuda(0).cpu().numpy())
print("x2张量转列表： ", x2.cuda(0).cpu().numpy().tolist())
print("x2 shape： ", x2.shape)
print("x2 size： ", x2.size())
print("x2 exp()： ", torch.exp(x2))
print("x2 sum dim=0:    ", torch.sum(x2, dim=0))
print("x2 sum dim=1:    ", torch.sum(x2, dim=1))
print("x2 转置transpose(1, 0):    ", x2.transpose(1, 0))
print(x2.view(3, 2))
print(x2.flatten())

