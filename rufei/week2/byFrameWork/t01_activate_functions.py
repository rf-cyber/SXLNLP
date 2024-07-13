# coding:utf-8
import torch

# 标量
# x = [-1, 0, 1, 2, 3, 4]

x = [[-1, 0, 1, 2, 3, 4],
     [-2, 1, 4, 0, -1, 2]]

# tensor张量
x_tensor = torch.Tensor(x)

# use_gpu = True
use_gpu = False

if use_gpu and torch.cuda.is_available():
    print("GPU is available")
    print(torch.cuda.device_count())  # 查看有多少个GPU  1
    print(torch.cuda.current_device())  # 查看当前GPU的编号  0
    print(torch.cuda.get_device_name(0))  # 查看当前GPU的名称  NVIDIA GeForce RTX 3060
    print(torch.version.cuda)  # 查看CUDA版本  12.1
    print(torch.cuda.get_device_capability(0))  # 查看当前GPU的计算能力  [8, 6]
    # 查看当前GPU的属性  {'name': 'NVIDIA GeForce RTX 3060', 'major': 8, 'minor': 6,
    # 'total_memory': 12288000000, 'multi_processor_count': 48}
    print(torch.cuda.get_device_properties(0))

    # 定义使用的设备（GPU）
    device = torch.device("cuda:0")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将张量移动到GPU上，同样可以使用.to()方法
    x_tensor = x_tensor.cuda("cuda:0")
    # x_tensor = x_tensor.cuda(0)
    # x_tensor = x_tensor.cuda(device)

    # 将张量移动到GPU上，同样可以使用.cuda()方法
    XX = x_tensor.to(device)
else:
    print("GPU is not available")
    XX = x_tensor


# 激活函数-sigmoid
y_sigmoid = torch.sigmoid(x_tensor)

# 激活函数-relu
y_relu = torch.relu(XX)

# 激活函数-tanh
y_tanh = torch.tanh(XX)

# 激活函数-softmax
y_softmax = torch.softmax(XX, dim=0)

print("x = ", x)
print("y_sigmoid = ", y_sigmoid)
print("y_tanh = ", y_tanh)
print("y_softmax = ", y_softmax)
print("y_softmax = ", y_softmax.cpu().detach().numpy())  # 将张量从GPU上移动到CPU上
