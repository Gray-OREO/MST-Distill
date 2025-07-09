import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, cls_num=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=cls_num)
        self.flatten_identity = nn.Identity()
        self.hook_names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2', 'flatten_identity']

    def forward(self, x):
        # 卷积层C1 + 激活函数 + 池化层S2
        x = self.pool1(F.relu(self.conv1(x)))
        # 卷积层C3 + 激活函数 + 池化层S4
        x = self.pool2(F.relu(self.conv2(x)))
        # 卷积层C5 + 激活函数
        x = F.relu(self.conv3(x))
        # 将特征图展平为一维向量
        x = x.view(-1, 120)
        x = self.flatten_identity(x)
        # 全连接层F6 + 激活函数
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x


class LeNet5_woClsHead(nn.Module):
    def __init__(self,):
        super(LeNet5_woClsHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.hook_names = ['conv1', 'conv2', 'fc1']

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        return x


if __name__ == '__main__':
    model = LeNet5()
    print(model)

    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    print(output)
