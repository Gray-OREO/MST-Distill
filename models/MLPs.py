import torch
import torch.nn as nn
import torch.nn.functional as F


class visualMLP(nn.Module):
    def __init__(self, cls_num=8):
        super(visualMLP, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, cls_num)
        self.hook_names = ['fc1', 'fc2', 'fc3']

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class textualMLP(nn.Module):
    def __init__(self, cls_num=8):
        super(textualMLP, self).__init__()
        self.fc1 = nn.Linear(768, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, cls_num)
        self.hook_names = ['fc1', 'fc2', 'fc3']

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class visualMLP_woClsHead(nn.Module):
    def __init__(self):
        super(visualMLP_woClsHead, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.hook_names = ['fc1', 'fc2']

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class textualMLP_woClsHead(nn.Module):
    def __init__(self):
        super(textualMLP_woClsHead, self).__init__()
        self.fc1 = nn.Linear(768, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.hook_names = ['fc1', 'fc2']

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class IntermediateFusionMLP(nn.Module):
    def __init__(self):
        super(IntermediateFusionMLP, self).__init__()
        self.VisualNet = visualMLP_woClsHead()
        self.TextualNet = textualMLP_woClsHead()
        self.fc1 = nn.Linear(512+512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 8)
        self.hook_names = ['VisualNet.fc1', 'VisualNet.fc2', 'TextualNet.fc1', 'TextualNet.fc2',
                           'fc1', 'fc2', 'fc3']

    def forward(self, x1, x2):
        x1 = self.VisualNet(x1)
        x2 = self.TextualNet(x2)
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    x1 = torch.randn(1, 2048)
    x2 = torch.randn(1, 768)

    # Tea.-MM-I
    model = IntermediateFusionMLP()
    y = model(x1, x2)

    # Stu.-V
    # model = visualMLP()
    # y = model(x1)

    # Stu.-T
    # model = textualMLP()
    # y = model(x2)

    print(y.size())
    print(y)