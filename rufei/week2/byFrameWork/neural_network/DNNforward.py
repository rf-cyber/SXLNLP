# coding:utf8

import torch
import torch.nn as nn

"""
numpy手动实现一个线性层:

线性层设计：

1. 输入x: 1*5
    w1: 10 * 5
    b1: 1*5
2. 输出hidden: 1*10
    w2: 10 * 10
    b2: 1*10
3. 输入hidden: 1*10
    w3: 10 * 3
    b3: 1*3
4. 输出y_pred: 1*3

"""


class TorchModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=10, output_size=3):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size).cuda(0)  # w1: 5 * 10, b1: 1*10
        self.layer2 = nn.Linear(hidden_size, hidden_size).cuda(0)  # w2: 10 * 10, b2: 1*10
        self.layer3 = nn.Linear(hidden_size, output_size).cuda(0)  # w3: 10 * 3, b3: 1*3

    def forward(self, x):
        x = self.layer1(x).cuda(0)  # shape: (batch_size, input_size) -> (batch_size, hidden_size1)
        x = self.layer2(x).cuda(0)  # shape: (batch_size, hidden_size1) -> (batch_size, hidden_size2)
        y_pred = self.layer3(x).cuda(0)  # shape: (batch_size, hidden_size2) -> (batch_size, output_size))
        return y_pred


if __name__ == '__main__':
    # 随便准备一个网络输入
    torch_x = torch.Tensor([3.1, 1.3, 1.2, 0.9, 0.7]).cuda(0)
    # 建立torch模型
    torch_model = TorchModel()

    print(torch_model.state_dict())

    print("-----------")
    # 打印模型权重，权重为随机初始化
    torch_model_w1 = torch_model.state_dict()["layer1.weight"].cpu().numpy()
    torch_model_b1 = torch_model.state_dict()["layer1.bias"].cpu().numpy()
    torch_model_w2 = torch_model.state_dict()["layer2.weight"].cpu().numpy()
    torch_model_b2 = torch_model.state_dict()["layer2.bias"].cpu().numpy()
    print(torch_model_w1, "torch w1 权重")
    print(torch_model_b1, "torch b1 权重")
    print("-----------")
    print(torch_model_w2, "torch w2 权重")
    print(torch_model_b2, "torch b2 权重")
    print("-----------")
    # 使用torch模型做预测
    y_pred = torch_model.forward(torch_x)
    print("torch模型预测结果：", y_pred)
