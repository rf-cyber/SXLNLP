import torch
import torch.nn as nn

print(torch.__version__)  # 1.9.0

print(torch.cuda.is_available())  # True

print(torch.cuda.current_device())  # 0

print(torch.cuda.get_device_name(0))  # NVIDIA GeForce GTX 1080 Ti
# 1.9.0 True 0 NVIDIA GeForce GTX 1080 Ti

# 初始化权重
embedding = nn.Embedding(6, 3).cuda(0)
print("初始化权重：\n", embedding.weight)  # shape 1 * 6 * 3
"""
tensor([[-0.6092, -0.5478, -0.3393],
        [-0.2713, -0.7802,  0.0314],
        [ 0.4565, -0.6125,  1.4503],
        [ 0.8967,  0.4512,  0.0768],
        [ 0.8492,  0.2650,  0.4339],
        [ 0.4074, -0.0119, -0.3705]], device='cuda:0', requires_grad=True)
"""

# 输入：每个元素表示在embedding中的位置
# [1,2,3,4] -> 表示取embedding中的第1，2，3，4行词向量，[[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
# 输入shape： batch_size * seq_len，此例中，batch_size省略了。
input1 = torch.LongTensor([0, 1, 2, 3]).cuda(0)
output1 = embedding(input1)
print("输出：\n", output1)  # shape 1 * 4 * 3
"""
tensor([[-0.6092, -0.5478, -0.3393],
        [-0.2713, -0.7802,  0.0314],
        [ 0.4565, -0.6125,  1.4503],
        [ 0.8967,  0.4512,  0.0768]], device='cuda:0', grad_fn=<EmbeddingBackward0>)
"""

# 输入：每个元素表示在embedding中的位置
# [1,2,3,4] -> 表示取embedding中的第1，2，3，4行词向量，[[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
# 输入shape： batch_size * seq_len，此例中，batch_size为1。
input0 = torch.LongTensor([[0, 1, 2, 3]]).cuda(0)
output0 = embedding(input0)
print("输出：\n", output0)  # shape 1 * 4 * 3
"""
tensor([[[-0.6092, -0.5478, -0.3393],
        [-0.2713, -0.7802,  0.0314],
        [ 0.4565, -0.6125,  1.4503],
        [ 0.8967,  0.4512,  0.0768]]], device='cuda:0', grad_fn=<EmbeddingBackward0>)
"""

input = torch.LongTensor([0, 1, 4, 5, 4, 3, 2, 5]).cuda(0)
output = embedding(input)
print("输出类型：\n", type(output))  # <class 'torch.Tensor'>
print("输出：\n", output)  # shape 1 * 8 * 3

input2 = torch.LongTensor([[0, 1, 4, 5, 4, 3, 2, 5]]).cuda(0)
output2 = embedding(input2)
print("输出：\n", output2)  # shape 1 * 1 * 8 * 3



