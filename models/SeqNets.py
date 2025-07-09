import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class LSTM(nn.Module):
    def __init__(self, cls_num, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size//2, num_layers=2, batch_first=True)
        self.fc = nn.Linear(input_size//2, cls_num)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 使用最后一个时间步的输出进行分类
        output = self.fc(lstm_out[:, -1, :])
        return output


class BiLSTM(nn.Module):
    def __init__(self, cls_num, input_size):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=input_size//2, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(input_size, cls_num)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.bilstm(x)
        # 使用最后一个时间步的输出进行分类
        output = self.fc(lstm_out[:, -1, :])
        return output


class GRU(nn.Module):
    def __init__(self, cls_num, input_size):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = input_size//2
        self.gru = nn.GRU(input_size=input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(input_size//2, cls_num)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class Video3DCNN(nn.Module):
    def __init__(self):
        super(Video3DCNN, self).__init__()
        # 3D卷积层
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=1)

        # 最大池化层
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # 全局平均池化，代替全连接层
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 最后的全连接层
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        # 输入x的维度为 (batch_size, channels, depth, height, width)
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 32, depth//2, height//2, width//2)
        x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 64, depth//4, height//4, width//4)
        x = self.pool(F.relu(self.conv3(x)))  # (batch_size, 128, depth//8, height//8, width//8)

        # 全局平均池化
        x = self.global_pool(x)  # (batch_size, 128, 1, 1, 1)
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 128)

        # 全连接层
        x = self.fc(x)  # 输出为 (batch_size, num_classes)
        return x


class AudioBranch(nn.Module):
    def __init__(self):
        super(AudioBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=15, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.hook_names = ['conv1', 'conv2', 'conv3']

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        return x


class VisualBranch(nn.Module):
    def __init__(self):
        super(VisualBranch, self).__init__()
        # 三层卷积
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(6)
        self.conv2 = nn.Conv3d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(12)
        self.conv3 = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(24)
        self.pool = nn.MaxPool3d(kernel_size=2)  # 使用池化层减小特征图尺寸
        # 全局平均池化，输出大小为 (batch_size, 320, 1, 1, 1)
        # self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.hook_names = ['conv1', 'conv2', 'conv3']

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 经过第一层卷积和池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 经过第二层卷积和池化
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 经过第三层卷积和池化
        # x = self.global_max_pool(x.squeeze(2))  # 全局平均池化降维
        x = x.view(x.size(0), -1)  # 展平，输出大小将是 (batch_size, 320)
        return x


class DualStreamCNN(nn.Module):
    def __init__(self, cls_num):
        super(DualStreamCNN, self).__init__()
        self.audio_branch = AudioBranch()
        self.visual_branch = VisualBranch()

        # MLP
        self.fc1 = nn.Linear(20096, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, cls_num)
        self.hook_names = ['visual_branch', 'visual_branch.conv1', 'visual_branch.conv2', 'visual_branch.conv3',
                           'audio_branch', 'audio_branch.conv1', 'audio_branch.conv2', 'audio_branch.conv3',
                           'fc1', 'fc2', 'fc3']

    def forward(self, x_visual, x_audio):
        audio_features = self.audio_branch(x_audio)  # 1280
        visual_features = self.visual_branch(x_visual)  # 18816

        # Concatenate audio and visual features
        combined_features = torch.cat((audio_features, visual_features), dim=1)

        # MLP for classification
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AudioBranchNet(nn.Module):
    def __init__(self, cls_num):
        super(AudioBranchNet, self).__init__()
        self.audio_branch = AudioBranch()

        # MLP
        self.fc1 = nn.Linear(1280, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, cls_num)
        self.hook_names = ['audio_branch', 'fc1', 'fc2', 'fc3']

    def forward(self, x_audio):
        audio_features = self.audio_branch(x_audio)

        # MLP for classification
        x = F.relu(self.fc1(audio_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VisualBranchNet(nn.Module):
    def __init__(self, cls_num):
        super(VisualBranchNet, self).__init__()
        self.visual_branch = VisualBranch()

        # MLP
        self.fc1 = nn.Linear(18816, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, cls_num)
        self.hook_names = ['visual_branch', 'fc1', 'fc2', 'fc3']

    def forward(self, x_visual):
        visual_features = self.visual_branch(x_visual)

        # MLP for classification
        x = F.relu(self.fc1(visual_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # 假设输入维度为 (batch_size, sequence_length, feature_dim)
    input_tensor_A = torch.randn(1, 15, 156)  # batch size 32, sequence length 15, feature dim 128
    input_tensor_V = torch.randn(1, 3, 15, 224, 224)

    # Tea.-MM DualStream-I
    # model = DualStreamCNN(cls_num=8)
    # summary(model, (input_tensor_A.shape, input_tensor_V.shape))
    # out = model(input_tensor_V, input_tensor_A)  # [bs, 8]

    # Stu.-A AudioBranchNet
    # model = AudioBranchNet(cls_num=8)
    # summary(model, input_tensor_A.shape)
    # out = model(input_tensor_A)  # [bs, 8]
    # print(out.shape)

    # Stu.-V VisualBranchNet
    # model = VisualBranchNet(cls_num=8)
    # summary(model, input_tensor_V.shape)
    # out = model(input_tensor_V)  # [bs, 8]
    # print(out.shape)
