import torch
import torch.nn as nn


# 定义交叉熵
ce_loss = nn.CrossEntropyLoss()

# 定义预测值 (batch_size, num_classes) # 每一行代表一个样本，每一列代表一个类别
prep = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.2, 0.2, 0.3],
                          [0.1, 0.3, 0.2]])
# 定义目标值
target = torch.LongTensor([0, 1, 1])

# 计算交叉熵
loss = ce_loss(prep, target)
