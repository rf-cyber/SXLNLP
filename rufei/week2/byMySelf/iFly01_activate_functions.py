# coding:utf8
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(sigmoid(x))
    x = np.array([[0.7, 0.2, 0.1]])
    print(softmax(x))
