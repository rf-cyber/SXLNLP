# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import copy

"""
基于pytorch的网络编写
加入激活函数
x = [x0, x1]
w = [[w11, w12], [w21, w22]]
y = [y0, y1]
"""


class TorchModel(nn.Module):
    def __init__(self, hidden_size):
        super(TorchModel, self).__init__()
        self.layer = nn.Linear(hidden_size, hidden_size, bias=False)  # w = hidden_size * hidden_size  wx+b -> wx
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss  # loss采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.layer(x)
        y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


x = np.array([-0.5, 0.1])  # 输入
y = np.array([0.1, 0.2])  # 预期输出

# torch实验
torch_model = TorchModel(2)
torch_model_w = torch_model.state_dict()["layer.weight"]
print(torch_model_w, "初始化权重")
numpy_model_w = copy.deepcopy(torch_model_w.numpy())
# numpy array -> torch tensor, unsqueeze的目的是增加一个batchsize维度
torch_x = torch.from_numpy(x).float().unsqueeze(0)
torch_y = torch.from_numpy(y).float().unsqueeze(0)
# torch的前向计算过程，得到loss
torch_loss = torch_model(torch_x, torch_y)
print("torch模型计算loss：", torch_loss)
# #手动实现loss计算


# # #设定优化器
learning_rate = 0.1
# optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(torch_model.parameters())
# optimizer.zero_grad()
# #
# # #pytorch的反向传播操作
torch_loss.backward()
# print(torch_model.layer.weight.grad, "torch 计算梯度")  #查看某层权重的梯度

# # #手动实现反向传播

# print(grad, "diy 计算梯度")
# #
# #torch梯度更新
optimizer.step()
# # #查看更新后权重
update_torch_model_w = torch_model.state_dict()["layer.weight"]
print(update_torch_model_w, "torch更新后权重")
