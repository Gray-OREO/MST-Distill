import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LeNet5 import LeNet5_woClsHead
from torchinfo import summary


class ThreeLayerCNN_A(nn.Module):
    def __init__(self, cls_num=10):
        super(ThreeLayerCNN_A, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 14 * 14, cls_num)
        self.conv3_maxpool = nn.Identity()
        self.conv3_maxpool_flatten = nn.Identity()
        self.hook_names = ['conv1', 'conv2', 'conv3', 'conv3_maxpool', 'conv3_maxpool_flatten', 'fc1']

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        conv3_maxpool = self.conv3_maxpool(x)
        x = conv3_maxpool.view(-1, 128 * 14 * 14)
        x = self.conv3_maxpool_flatten(x)
        x = self.fc1(x)
        return x


class FiveLayerCNN_A(nn.Module):
    def __init__(self):
        super(FiveLayerCNN_A, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc = nn.Linear(512 * 3 * 3, 10)  # Assuming input size is 112x112
        self.hook_names = ['conv1', 'conv3', 'conv5']

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 512 * 3 * 3)  # Flatten the tensor
        x = self.fc(x)
        return x


class FiveLayerCNN_A_woClsHead(nn.Module):
    def __init__(self):
        super(FiveLayerCNN_A_woClsHead, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.hook_names = ['conv1', 'conv3', 'conv5']

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = self.avg_pool(x).view(-1, 512)
        return x


class IntermediateFusionNet(nn.Module):
    def __init__(self):
        super(IntermediateFusionNet, self).__init__()
        self.AudioNet = FiveLayerCNN_A_woClsHead()
        self.ImageNet = LeNet5_woClsHead()
        self.fc1 = nn.Linear(596, 298)
        self.fc2 = nn.Linear(298, 10)
        self.hook_names = ['AudioNet.conv1', 'AudioNet.conv3', 'AudioNet.conv5',
                           'ImageNet.conv1', 'ImageNet.conv2','ImageNet.fc1',
                           'concat_identity', 'fc1', 'fc2']
        self.concat_identity = nn.Identity()

    def forward(self, x1, x2):
        x1 = self.ImageNet(x1)
        x2 = self.AudioNet(x2)
        concat_feature = self.concat_identity(torch.cat((x1, x2), dim=1))
        x = self.fc1(concat_feature)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    from utils import seed_all
    seed_all(0)
    # 示例输入：28x28的灰度图 112x112的mel频谱图
    input_tensor1 = torch.randn(1, 1, 28, 28)
    input_tensor2 = torch.randn(1, 1, 112, 112)

    # Tea.-MM L
    # model = LateFusionNet()
    # summary(model, (input_tensor1.shape, input_tensor2.shape))
    # res = model(input_tensor1, input_tensor2)  # [bs, 10]
    # print(res)

    # Tea.-MM I
    # model = IntermediateFusionNet()
    # summary(model, (input_tensor1.shape, input_tensor2.shape))
    # res = model(input_tensor1, input_tensor2)  # [bs, 10]
    # print(res)

    # Stu.-V
    # model = LeNet5()
    # summary(model, input_tensor1.shape)
    # res = model(input_tensor1)  # [bs, 10]

    # Stu.-A
    # model = ThreeLayerCNN_A()
    # summary(model, input_tensor2.shape)
    # res = model(input_tensor2)  # [bs, 10]