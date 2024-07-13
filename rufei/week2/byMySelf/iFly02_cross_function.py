import torch
import numpy as np

from rufei.week2.byMySelf.iFly01_activate_functions import softmax


def cross_entropy(predict, target):
    """
    交叉熵损失函数
    :param predict:
    :param target:
    :return: batch_size * entropy
    """
    return -np.sum(target * np.log(predict), axis=1) / len(predict)


def cross_entropy2(predict, target):
    """
    交叉熵损失函数
    :param predict:
    :param target:
    :return: int, 所有样本的交叉熵之平均和
    """
    batch_size, _ = predict.shape
    predict = softmax(predict)
    target = to_one_hot(target, predict.shape)
    entropy = -np.sum(target * np.log(predict), axis=1)
    return sum(entropy) / batch_size


def to_one_hot(target, shape):
    one_hot = np.zeros(shape)
    for idx, t_type in enumerate(target):
        one_hot[idx][t_type] = 1
    return one_hot


def mean_squared_error(predict, target):
    """
    均方误差损失函数
    :param predict:
    :param target:
    :return: int, 所有样本的均方误差之和
    """

    target = to_one_hot(target, predict.shape)
    return np.mean((predict - target) ** 2)


def squared_error_batch(predict, target):
    """
    方差损失函数
    :param predict:
    :param target:
    :return:
    """
    target = to_one_hot(target, predict.shape)
    return np.sum((predict - target) ** 2)


def squared_error_single(predict, target):
    """
    方差损失函数
    :param predict:
    :param target:
    :return:
    """
    return (predict - target) ** 2


def squared_error_batch_torch(predict, target):
    """
    方差损失函数
    :param predict:
    :param target:
    :return:
    """
    return torch.sum((predict - target) ** 2)


if __name__ == '__main__':
    predict = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
    target = np.array([0, 2])
