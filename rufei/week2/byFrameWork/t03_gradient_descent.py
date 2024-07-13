import torch
import matplotlib.pyplot as plt

from rufei.week2.byMySelf.iFly02_cross_function import squared_error_batch_torch

# 梯度下降
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor(2.)
x.grad.zero_()
y = x ** 3
y.backward()
print(x.grad)  # tensor(3.)

# 初始化权重
w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)


# 定义模型
def model(x):
    return w1 * x ** 2 + w2 * x + b


def init_data():
    """
    初始化数据集
    :return:
    """
    x_train = [0.01 * x for x in range(100)]
    y_train = [2 * x ** 2 + 3 * x + 4 for x in x_train]

    return x_train, y_train


def predict(x):
    """
    预测函数
    :param x:
    :return:
    """
    return [model(i) for i in x]


def train():
    for epoch in range(100):
        x_train, y_train = init_data()
        x = torch.tensor(x_train, requires_grad=True)
        y_target = torch.tensor(y_train)
        y_pred = model(x)
        # loss = torch.sum((y_pred - y_target) ** 2)
        loss = squared_error_batch_torch(y_pred, y_target)
        loss.backward()
        # 更新参数
        w1.data -= 0.01 * w1.grad
        w2.data -= 0.01 * w2.grad
        b.data -= 0.01 * b.grad
        # 清空梯度
        w1.grad.zero_()
        w2.grad.zero_()
        b.grad.zero_()

        if epoch % 10 == 0:
            print(f'epoch: {epoch}, loss: {loss.item()}')
            print(f'w1: {w1.item()}, w2: {w2.item()}, b: {b.item()}')
        if loss.item() < 1e-3:
            break

    print(f'w1: {w1.item()}, w2: {w2.item()}, b: {b.item()}')
    print(w1, w2, b)


"""
# 注意：w.grad.zero_() 和 w.grad.zero() 的区别
# zero_() 会直接在原地修改，而 zero() 会返回一个新的 tensor
# 所以在更新参数时，需要使用 zero_()
# w.grad.zero() 会报错，因为 zero() 返回的是一个新的 tensor，而不是原来的 tensor
# 所以在更新参数时，需要使用 zero_()
# w.grad.zero_() 不会报错，因为 zero_() 会直接在原地修改，所以返回的是原来的 tensor
# 所以在更新参数时，需要使用 zero_()
"""

if __name__ == '__main__':
    train()
    print(f"训练完成！y = {w1} * x + {b}")
    x_test = [2 * x for x in range(100)]
    y_test = [2 * x ** 2 + 3 * x + 4 for x in x_test]
    y_pred = predict(x_test)

    print(y_pred)
    print(torch.Tensor(y_pred).numpy())

    # 根据用途可分为：
    # 回归模型：根据输入，预测输出值
    # 分类模型：根据输入，预测输出类别（二分类：0，1；多分类：0，1，2，3，4，5，6，7，8，9），预测值大于0.5的为1，小于0.5的为0。
    plt.plot(x_test, y_test, label='real')
    plt.plot(x_test, torch.Tensor(y_pred).numpy(), label='predict', color='red')
    plt.show()
