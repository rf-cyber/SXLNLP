# coding:utf8

import numpy as np

"""
手动实现梯度计算和反向传播
加入激活函数
x = [x0, x1]
w = [[w11, w12], [w21, w22]]
y = [y0, y1]
"""


# 自定义模型，接受一个参数矩阵作为入参
class DiyModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        x = np.dot(x, self.weight.T)
        y_pred = self.diy_sigmoid(x)
        if y is not None:
            return self.diy_mse_loss(y_pred, y)
        else:
            return y_pred

    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'forward'):
            raise AttributeError('Model has no forward method')
        return self.forward(*args, **kwargs)

    # sigmoid
    def diy_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 手动实现mse，均方差loss
    def diy_mse_loss(self, y_pred, y_true):
        return np.sum(np.square(y_pred - y_true)) / len(y_pred)

    # 手动实现梯度计算
    def calculate_grad(self, y_pred, y_true, x):
        # 前向过程
        # wx = np.dot(self.weight, x)
        # sigmoid_wx = self.diy_sigmoid(wx)
        # loss = self.diy_mse_loss(sigmoid_wx, y_true)
        # 反向过程
        # 均方差函数 (y_pred - y_true) ^ 2 / n 的导数 = 2 * (y_pred - y_true) / n , 结果为2维向量
        grad_mse = 2 / len(x) * (y_pred - y_true)
        # sigmoid函数 y = 1/(1+e^(-x)) 的导数 = y * (1 - y), 结果为2维向量
        grad_sigmoid = y_pred * (1 - y_pred)
        # wx矩阵运算，见ppt拆解, wx = [w11*x0 + w21*x1, w12*x0 + w22*x1]
        # 导数链式相乘
        grad_w11 = grad_mse[0] * grad_sigmoid[0] * x[0]
        grad_w12 = grad_mse[1] * grad_sigmoid[1] * x[0]
        grad_w21 = grad_mse[0] * grad_sigmoid[0] * x[1]
        grad_w22 = grad_mse[1] * grad_sigmoid[1] * x[1]
        grad = np.array([[grad_w11, grad_w12],
                         [grad_w21, grad_w22]])
        # 由于pytorch存储做了转置，输出时也做转置处理
        return grad.T

    # sgd梯度更新
    def diy_sgd(grad, weight, learning_rate):
        return weight - learning_rate * grad

    # adam梯度更新
    def diy_adam(grad, weight):
        # 参数应当放在外面，此处为保持后方代码整洁简单实现一步
        alpha = 1e-3  # 学习率
        beta1 = 0.9  # 超参数
        beta2 = 0.999  # 超参数
        eps = 1e-8  # 超参数
        t = 0  # 初始化
        mt = 0  # 初始化
        vt = 0  # 初始化
        # 开始计算
        t = t + 1
        gt = grad
        mt = beta1 * mt + (1 - beta1) * gt
        vt = beta2 * vt + (1 - beta2) * gt ** 2
        mth = mt / (1 - beta1 ** t)
        vth = vt / (1 - beta2 ** t)
        weight = weight - (alpha * mth / (np.sqrt(vth) + eps))
        return weight


if __name__ == '__main__':
    # 设定优化器
    learning_rate = 0.1

    print("初始化输入...")
    x = np.array([-0.5, 0.1])  # 输入
    y = np.array([0.1, 0.2])  # 预期输出

    print("初始化权重...")
    np.random.seed(0)
    numpy_model_w = np.random.randn(2, 2)

    # 手动实现loss计算
    print("手动实现loss计算...")
    diy_model = DiyModel(numpy_model_w)
    diy_loss = diy_model.forward(x, y)
    print("diy模型计算loss：", diy_loss)

    # 手动实现反向传播
    grad = diy_model.calculate_grad(diy_model.forward(x), y, x)
    print("diy 计算梯度: ", grad)

    # 手动梯度更新
    # diy_update_w = DiyModel.diy_sgd(grad, numpy_model_w, learning_rate)
    diy_update_w = DiyModel.diy_adam(grad, numpy_model_w)
    print("diy更新权重: ", diy_update_w)
