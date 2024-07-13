# coding:utf8

import numpy as np

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


class TorchModel:
    def __init__(self, input_size=5, hidden_size=10, output_size=3, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # 初始化权重
        self.w1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.random.randn(1, hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size)
        self.b2 = np.random.randn(1, hidden_size)
        self.w3 = np.random.randn(output_size, hidden_size)
        self.b3 = np.random.randn(1, output_size)

        print("w1:\n", self.w1)
        print("b1:\n", self.b1)
        print("w2:\n", self.w2)
        print("b2:\n", self.b2)
        print("w3:\n", self.w3)
        print("b3:\n", self.b3)

    def forward(self, x):
        pass

    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'forward'):
            raise NotImplementedError('Model has no forward method')
        return self.forward(*args, **kwargs)

    def linear(self, x):
        pass


class DiyModel(TorchModel):
    def __init__(self, ):
        super().__init__(seed=10)

    def forward(self, x):
        hidden1 = np.dot(x, self.w1.T) + self.b1  # 1*5
        hidden2 = np.dot(hidden1, self.w2.T) + self.b2  # 1*10
        y_pred = np.dot(hidden2, self.w3.T) + self.b3  # 1*3
        return y_pred


if __name__ == '__main__':
    # #把torch模型权重拿过来自己实现计算过程
    diy_model = DiyModel()
    # 准备网络输入
    x = np.array([3.1, 1.3, 1.2, 0.9, 0.7])
    # #用自己的模型来预测
    y_pred_diy = diy_model(x)
    print("diy模型预测结果：", y_pred_diy)
    print("")
