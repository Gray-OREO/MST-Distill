import torch
import torch.nn as nn
from torchinfo import summary

def get_HTmodules(args, model_s):
    regressors = []
    hint_guided_names = [[],[]]
    if args.database == 'AV-MNIST':
        if args.Tmodel == 'CNN-I':
            if args.Smodel == 'LeNet5':
                hint_guided_names = [['concat_identity', 'fc1'], ['conv3', 'fc1']]
                regressor1 = Regressor_2D_1D(input_shape=(120, 1, 1), target_length=596)
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=84, target_length=298)
                regressors.append(regressor2)
            elif args.Smodel == 'ThreeLayerCNN-A':
                hint_guided_names = [['concat_identity', 'fc1'], ['conv2', 'conv3']]
                regressor1 = Regressor_2D_1D(input_shape=(64, 56, 56), target_length=596)
                regressors.append(regressor1)
                regressor2 = Regressor_2D_1D(input_shape=(128, 28, 28), target_length=298)
                regressors.append(regressor2)
        elif args.Tmodel == 'LeNet5':
            if args.Smodel == 'ThreeLayerCNN-A':
                hint_guided_names = [['conv1', 'flatten_identity'], ['conv2', 'conv3_maxpool_flatten']]
                regressor1 = Regressor_2D_2D_DownSampling(input_shape=(64, 56, 56), target_shape=(6, 28, 28))
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=25088, target_length=120)
                regressors.append(regressor2)
        elif args.Tmodel == 'ThreeLayerCNN-A':
            if args.Smodel == 'LeNet5':
                hint_guided_names = [['conv2', 'conv3_maxpool_flatten'], ['conv1', 'flatten_identity']]
                regressor1 = Regressor_2D_2D_UpSampling(input_shape=(6, 28, 28), target_shape=(64, 56, 56))
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=120, target_length=25088)
                regressors.append(regressor2)

    elif args.database == 'NYU-Depth-V2':
        if args.Tmodel == 'FuseNet-I':
            if args.Smodel == 'FuseNet-RGBbranch':
                hint_guided_names = [['after_fusion_identity', 'CBR5_RGBD_DEC'], ['CBR5_RGB_ENC', 'CBR5_RGB_DEC']]
                regressor1 = Regressor_2D_2D_DownSampling(input_shape=(512, 30, 40), target_shape=(512, 15, 20))
                regressors.append(regressor1)
                regressor2 = Regressor_2D_2D_resoFixed(input_shape=(512, 30, 40), target_shape=(512, 30, 40))
                regressors.append(regressor2)
            if args.Smodel == 'FuseNet-Dbranch':
                hint_guided_names = [['after_fusion_identity', 'CBR5_RGBD_DEC'], ['CBR5_DEPTH_ENC', 'CBR5_D_DEC']]
                regressor1 = Regressor_2D_2D_DownSampling(input_shape=(512, 30, 40), target_shape=(512, 15, 20))
                regressors.append(regressor1)
                regressor2 = Regressor_2D_2D_resoFixed(input_shape=(512, 30, 40), target_shape=(512, 30, 40))
                regressors.append(regressor2)
        elif args.Tmodel == 'FuseNet-RGBbranch':
            if args.Smodel == 'FuseNet-Dbranch':
                hint_guided_names = [['CBR4_RGB_ENC', 'CBR4_RGB_DEC'], ['CBR4_DEPTH_ENC', 'CBR4_D_DEC']]
                regressor1 = Regressor_2D_2D_resoFixed(input_shape=(512, 60, 80), target_shape=(512, 60, 80))
                regressors.append(regressor1)
                regressor2 = Regressor_2D_2D_resoFixed(input_shape=(256, 60, 80), target_shape=(256, 60, 80))
                regressors.append(regressor2)
        elif args.Tmodel == 'FuseNet-Dbranch':
            if args.Smodel == 'FuseNet-RGBbranch':
                hint_guided_names = [['CBR4_DEPTH_ENC', 'CBR4_D_DEC'], ['CBR4_RGB_ENC', 'CBR4_RGB_DEC']]
                regressor1 = Regressor_2D_2D_resoFixed(input_shape=(512, 60, 80), target_shape=(512, 60, 80))
                regressors.append(regressor1)
                regressor2 = Regressor_2D_2D_resoFixed(input_shape=(256, 60, 80), target_shape=(256, 60, 80))
                regressors.append(regressor2)

    elif args.database == 'RAVDESS':
        if args.Tmodel == 'DSCNN-I':
            if args.Smodel in ['AudioBranchNet', 'VisualBranchNet']:
                hint_guided_names = [['fc1', 'fc2'], ['fc1', 'fc2']]
                regressor1 = Regressor_1D_1D(input_length=320, target_length=320)
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=160, target_length=160)
                regressors.append(regressor2)
        elif args.Tmodel == 'VisualBranchNet':
            if args.Smodel == 'AudioBranchNet':
                hint_guided_names = [['fc1', 'fc2'], ['fc1', 'fc2']]
                regressor1 = Regressor_1D_1D(input_length=320, target_length=320)
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=160, target_length=160)
                regressors.append(regressor2)
        elif args.Tmodel == 'AudioBranchNet':
            if args.Smodel == 'VisualBranchNet':
                hint_guided_names = [['fc1', 'fc2'], ['fc1', 'fc2']]
                regressor1 = Regressor_1D_1D(input_length=320, target_length=320)
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=160, target_length=160)
                regressors.append(regressor2)

    elif args.database == 'VGGSound-50k':
        if args.Tmodel == 'DSCNN-VGGS-I':
            if args.Smodel in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
                hint_guided_names = [['fc1', 'fc2'], ['fc1', 'fc2']]
                regressor1 = Regressor_1D_1D(input_length=320, target_length=320)
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=160, target_length=160)
                regressors.append(regressor2)
        elif args.Tmodel == 'VisualBranchNet-VGGS':
            if args.Smodel == 'AudioBranchNet-VGGS':
                hint_guided_names = [['fc1', 'fc2'], ['fc1', 'fc2']]
                regressor1 = Regressor_1D_1D(input_length=320, target_length=320)
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=160, target_length=160)
                regressors.append(regressor2)
        elif args.Tmodel == 'AudioBranchNet-VGGS':
            if args.Smodel == 'VisualBranchNet-VGGS':
                hint_guided_names = [['fc1', 'fc2'], ['fc1', 'fc2']]
                regressor1 = Regressor_1D_1D(input_length=320, target_length=320)
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=160, target_length=160)
                regressors.append(regressor2)

    elif args.database == 'CMMD-V2':
        if args.Tmodel == 'MLP-I':
            if args.Smodel in ['MLP-Vb', 'MLP-Tb']:
                hint_guided_names = [['fc1', 'fc2'], ['fc1', 'fc2']]
                regressor1 = Regressor_1D_1D(input_length=1024, target_length=512)
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=512, target_length=256)
                regressors.append(regressor2)
        elif args.Tmodel in ['MLP-Vb', 'MLP-Tb']:
            if args.Smodel in ['MLP-Vb', 'MLP-Tb']:
                hint_guided_names = [['fc1', 'fc2'], ['fc1', 'fc2']]
                regressor1 = Regressor_1D_1D(input_length=1024, target_length=1024)
                regressors.append(regressor1)
                regressor2 = Regressor_1D_1D(input_length=512, target_length=512)
                regressors.append(regressor2)

    criterion = nn.MSELoss()
    print("Params to guide:",
          [name for name, param in model_s.named_parameters() if any(layer in name for layer in hint_guided_names[1])])
    params_to_guide = [param for name, param in model_s.named_parameters() if
                       any(layer in name for layer in hint_guided_names[1])]
    params = [{'params': net.parameters()} for net in regressors]
    params = params + [{'params': params_to_guide}]
    optim = torch.optim.Adam(params, lr=args.hint_lr)
    return hint_guided_names, regressors, criterion, optim


def hint_training_loss(hint_guided_names, regressors, criterion, features_t, features_s):
    loss = 0
    for i, regressor in enumerate(regressors):
        hint_feat = hint_feature_extract(hint_guided_names[0][i], features_t)
        guided_feat = feature_check(features_s[hint_guided_names[1][i]])
        output = regressor(guided_feat)
        loss += criterion(output, hint_feat)
    return loss


def hint_feature_extract(hint_name, features):
    hint_feature = features[hint_name]
    if isinstance(hint_feature, list):  # 检查值是否为列表
        if len(hint_feature) == 2:
            hint_feature = torch.cat((features[hint_name][0], features[hint_name][1]), dim=1)
        else:
            hint_feature = features[hint_name][0]
    if isinstance(hint_feature, tuple):  # 检查值是否为列表
        if len(hint_feature) == 2:
            hint_feature = torch.cat((features[hint_name][0], features[hint_name][1]), dim=2)
        if len(hint_feature) == 3:
            hint_feature = hint_feature[0]
    return hint_feature.detach()


def regressors_train(regressors):
    for regressor in regressors:
        regressor.train()


def regressors_eval(regressors):
    for regressor in regressors:
        regressor.eval()


# 特征自检函数
def feature_check(value):
    # 检查是否为列表
    if isinstance(value, list) and len(value) == 1:
        return value[0]  # 解包并返回列表的第一个元素
    return value  # 如果不是列表或长度不为1，返回原值


class Regressor_1D_1D(nn.Module):
    def __init__(self, input_length, target_length=20):
        super(Regressor_1D_1D, self).__init__()

        self.target_length = target_length  # 目标输出长度
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        # 计算需要的stride来匹配目标大小
        self.calculate_stride(input_length)

    def calculate_stride(self, input_length):
        """
        计算需要的stride和padding以使输入大小匹配目标输出大小
        """
        stride = max(input_length // self.target_length, 1)
        self.conv1d.stride = (stride,)

    def forward(self, x):
        # x的形状应为 (batch_size, seq_len)
        x = x.unsqueeze(1)  # 转换为 (batch_size, 1, seq_len)，符合 Conv1d 的输入
        x = self.conv1d(x)
        x = torch.nn.functional.interpolate(x, size=self.target_length)  # 调整到目标大小
        return x.squeeze(1)  # 转换回 (batch_size, target_length)


class Regressor_2D_1D(nn.Module):
    def __init__(self, input_shape, target_length):
        """
        input_shape: 输入的形状 (channels, height, width)
        target_length: 目标输出长度
        """
        super(Regressor_2D_1D, self).__init__()

        self.target_length = target_length

        # 定义2维卷积层
        self.conv2d = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)

        # 计算卷积后输出的大小，作为全连接层的输入大小
        conv_output_size = self._get_conv_output(input_shape)

        # 将卷积特征展平并通过全连接层映射到目标长度
        self.fc = nn.Linear(conv_output_size, target_length)

    def _get_conv_output(self, shape):
        """
        通过一次前向传播来计算卷积层的输出大小
        """
        x = torch.rand(1, *shape)  # 模拟输入 (batch_size=1, channels, height, width)
        x = self.conv2d(x)
        return x.numel()  # 返回展平后的特征维度大小

    def forward(self, x):
        # x 的形状为 (batch_size, channels, height, width)
        x = self.conv2d(x)  # 通过2维卷积
        x = x.view(x.size(0), -1)  # 展平 (batch_size, flattened_features)
        x = self.fc(x)  # 全连接层映射到 (batch_size, target_length)
        return x


class Regressor_2D_2D_resoFixed(nn.Module):
    def __init__(self, input_shape, target_shape):
        """
        input_shape: 输入的形状 (channels, height, width)
        target_shape: 目标输出形状 (target_height, target_width)
        """
        super(Regressor_2D_2D_resoFixed, self).__init__()

        # Depthwise 卷积：每个输入通道使用一个独立的卷积核
        self.conv = nn.Conv2d(input_shape[0], target_shape[0], kernel_size=3,
                                   stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)  # 先执行 Depthwise 卷积
        return x


class Regressor_2D_2D_DownSampling(nn.Module):
    def __init__(self, input_shape, target_shape):
        """
        input_shape: 输入的形状 (channels, height, width)
        target_shape: 目标输出形状 (target_height, target_width)
        """
        super(Regressor_2D_2D_DownSampling, self).__init__()

        # 使用 stride=2 的卷积层实现下采样
        self.conv = nn.Conv2d(
            in_channels=input_shape[0],  # 输入通道数
            out_channels=target_shape[0],  # 输出通道数
            kernel_size=3,  # 卷积核大小
            stride=2,  # 下采样步长
            padding=1  # 确保下采样后形状大致匹配
        )

    def forward(self, x):
        x = self.conv(x)  # 下采样卷积
        return x


class Regressor_2D_2D_UpSampling(nn.Module):
    def __init__(self, input_shape, target_shape):
        """
        input_shape: 输入的形状 (channels, height, width)
        target_shape: 目标输出形状 (target_height, target_width)
        """
        super(Regressor_2D_2D_UpSampling, self).__init__()

        # 使用转置卷积实现上采样
        self.deconv = nn.ConvTranspose2d(
            in_channels=input_shape[0],  # 输入通道数
            out_channels=target_shape[0],  # 输出通道数
            kernel_size=3,  # 卷积核大小
            stride=2,  # 上采样步长
            padding=1,  # 填充确保输出形状匹配
            output_padding=1  # 解决输出尺寸对齐问题
        )

    def forward(self, x):
        x = self.deconv(x)  # 上采样
        return x


class Regressor_1D_1Dseq(nn.Module):
    def __init__(self, feature_len, target_length, num_timesteps):
        super(Regressor_1D_1Dseq, self).__init__()

        self.num_timesteps = num_timesteps

        # 第一层全连接层：将输入特征映射到更高维度
        self.fc1 = nn.Linear(feature_len, 128)
        self.fc2 = nn.Linear(128, 256)

        # 最后一层全连接层，将特征映射到目标维度
        self.fc3 = nn.Linear(256, target_length * num_timesteps)

    def forward(self, x):
        # x 的形状应为 (batch_size, feature_len)
        batch_size, feature_len = x.shape
        # 逐步扩展特征维度
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        # 最后一个全连接层将特征转换为 (batch_size, num_timesteps * target_length)
        x = self.fc3(x)
        # 调整形状为 (batch_size, num_timesteps, target_length)
        x = x.view(batch_size, self.num_timesteps, -1)
        return x


if __name__ == '__main__':
    feat_s1 = torch.randn(1, 25088)
    feat_s2 = torch.randn(1, 64, 56, 56)
    feat_t = torch.randn(1, 596)

    regressor1 = Regressor_1D_1D(input_length=25088, target_length=120)
    # regressor2 = Regressor_2D_1D(input_shape=(120, 1, 1), target_length=596)
    regressor3 = Regressor_2D_2D_DownSampling(input_shape=(64, 56, 56), target_shape=(6, 28, 28))

    # res1 = regressor1(feat_s1)
    # res2 = regressor2(feat_s2)
    # res3 = regressor3(feat_s2)

    # print(res1.shape)
    # print(res2.shape)
    # print(res3.shape)

    # feat_s = torch.randn(1, 256, 120, 160)
    # feat_t = torch.randn(1, 512, 120, 160)
    # regressor3 = Regressor_2D_2D_resoFixed(input_shape=(256, 120, 160), target_shape=(512, 120, 160))
    # res3 = regressor3(feat_s)
    # print(res3.shape)

    # feat_s3 = torch.randn(1, 320)
    # regressor4 = Regressor_1D_1Dseq(feature_len=320, target_length=512, num_timesteps=10)
    # res4 = regressor4(feat_s3)
    # print(res4.shape)

    # summary(regressor1, (1, 25088))
    # summary(regressor3, (1, 64, 56, 56))