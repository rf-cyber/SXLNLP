# coding:utf8

import torch
import numpy as np
import copy

from rufei.week3.byFrameWork.t04_backpropagation_optim import TorchModel
from rufei.week3.byMySelf.t04_backpropagation_optim import DiyModel

"""
基于pytorch的网络编写
手动实现梯度计算和反向传播
加入激活函数
加入优化器
x = [x0, x1]
w = [[w11, w12], [w21, w22]]
y = [y0, y1]
"""

# 设定优化器 sgd优化器使用
# learning_rate = 0.1

x = np.array([-0.5, 0.1])  # 输入
y = np.array([0.1, 0.2])  # 预期输出

# torch实验
torch_model = TorchModel(2)
torch_model_w = torch_model.state_dict()["layer.weight"]
print("初始化权重: ", torch_model_w)

# numpy array -> torch tensor,
torch_x = torch.from_numpy(x).float().unsqueeze(0)  # unsqueeze的目的是增加一个batchsize维度
torch_y = torch.from_numpy(y).float().unsqueeze(0)
# torch的前向计算过程，得到loss
torch_loss = torch_model(torch_x, torch_y)
print("torch模型计算loss：", torch_loss)

# =========================================
# 手动实现loss计算
numpy_model_w = copy.deepcopy(torch_model_w.numpy())  # 保持初始权重一致
diy_model = DiyModel(numpy_model_w)
diy_loss = diy_model.forward(x, y)
print("diy模型计算loss：", diy_loss)

# optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(torch_model.parameters())
# optimizer.zero_grad()

# pytorch的反向传播操作
torch_loss.backward()
print(torch_model.layer.weight.grad, "torch 计算梯度")  # 查看某层权重的梯度

# torch梯度更新
optimizer.step()
# 查看更新后权重
update_torch_model_w = torch_model.state_dict()["layer.weight"]
print("torch更新后权重: ", update_torch_model_w)

# =========================================
# 手动实现反向传播
grad = diy_model.calculate_grad(diy_model.forward(x), y, x)
print("diy 计算梯度: ", grad)

# 手动梯度更新
# diy_update_w = DiyModel.diy_sgd(grad, numpy_model_w, learning_rate)
diy_update_w = DiyModel.diy_adam(grad, numpy_model_w)
print("diy更新权重: ", diy_update_w)
