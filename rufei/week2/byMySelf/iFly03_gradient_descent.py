from rufei.week2.byMySelf.iFly02_cross_function import squared_error_single

import numpy as np
import matplotlib.pyplot as plt

# 初始化权重
w1 = 1.0
w2 = 0.0
b = -1

# 定义学习率
lr = 0.01
# 定义loss目标
loss_target = 0.0001
# 定义epoch_number
epoch_number = 1000
# 定义batch_size
batch_size = 20


# 定义模型
def model(x):
    y = w1 * x ** 2 + w2 * x + b
    return y


def init_data():
    """
    初始化数据集
    :return:
    """
    x_train = [0.01 * x for x in range(100)]
    y_train = [2 * x ** 2 + 3 * x + 4 for x in x_train]

    return x_train, y_train


# 训练模型---回归模型
def train():
    """
    训练模型
    :return:
    """
    global w1, w2, b, x
    # 训练数据
    x_train, y_train = init_data()

    for epoch in range(epoch_number):
        epoch_loss = 0
        for x, y in zip(x_train, y_train):
            """
            每训练一个样本，计算一次梯度，更新一次权重，计算损失值
            """
            # 模型训练
            y_pred = model(x)

            # 计算损失函数
            epoch_loss += squared_error_single(y_pred, y)

            # 计算梯度
            grad_w1 = 2 * (y_pred - y) * x ** 2
            grad_w2 = 2 * (y_pred - y) * x
            grad_b = 2 * (y_pred - y)

            # 更新权重
            w1 = w1 - lr * np.mean(grad_w1)
            w2 = w2 - lr * np.mean(grad_w2)
            b = b - lr * np.mean(grad_b)

        print("epoch:{},loss:{}".format(epoch, epoch_loss / len(x_train)))
        print("w1:{},w2:{},b:{}".format(w1, w2, b))

        #  每个epoch检查一下损失值，直到损失函数收敛
        if epoch_loss / len(x_train) < loss_target:
            break

    print("w1:{},w2:{},b:{}".format(w1, w2, b))


def predict(x):
    """
    预测函数
    :param x:
    :return:
    """
    return [model(i) for i in x]


if __name__ == "__main__":
    train()

    x_test = [0.01 * x for x in range(100)]
    y_test = [2 * x ** 2 + 3 * x + 4 for x in x_test]
    y_pred = predict(x_test)

    print(y_pred)

    # 根据用途可分为：
    # 回归模型：根据输入，预测输出值
    # 分类模型：根据输入，预测输出类别（二分类：0，1；多分类：0，1，2，3，4，5，6，7，8，9），预测值大于0.5的为1，小于0.5的为0。

    # 1. 可视化
    plt.scatter(x_test, y_pred, c="r")
    plt.scatter(x_test, y_test)
    plt.show()
