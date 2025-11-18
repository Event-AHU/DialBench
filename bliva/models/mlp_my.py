import torch
import torch.nn as nn
class MLPTransformer(nn.Module):
    def __init__(self):
        super(MLPTransformer, self).__init__()
        self.fc1 = nn.Linear(4224, 1408)  # [B, 1530, 4224] -> [B, 1530, 1408]
        self.pool = nn.AdaptiveAvgPool1d(256)  # 对序列长度1530进行池化

    def forward(self, x):
        # 输入 x: [1, 1530, 4224]
        x = self.fc1(x)  # [1, 1530, 1408]
        # 注意：pool 要求输入为 [B, C, T]，所以需要转置
        x = x.transpose(1, 2)  # [1, 1408, 1530]
        x = self.pool(x)  # [1, 1408, 255]
        x = x.transpose(1, 2)  # [1, 255, 1408]
        return x
