import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def get_C2KDmodules(args):
    if args.database == 'AV-MNIST':
        if args.Tmodel == 'CNN-I':
            px_t = ProxyNet1D(num_classes=10, input_dim=596)
            if args.Smodel == 'LeNet5':
                feat_names = [['concat_identity'], ['fc1']]
                px_s = ProxyNet1D(num_classes=10, input_dim=84)
            elif args.Smodel == 'ThreeLayerCNN-A':
                feat_names = [['concat_identity'], ['conv3_maxpool']]
                px_s = ProxyNet(num_classes=10, in_channels=128)
        elif args.Tmodel == 'LeNet5':
            px_t = ProxyNet1D(num_classes=10, input_dim=84)
            feat_names = [['fc1'], ['conv3_maxpool']]
            px_s = ProxyNet(num_classes=10, in_channels=128)
        elif args.Tmodel == 'ThreeLayerCNN-A':
            px_t = ProxyNet(num_classes=10, in_channels=128)
            feat_names = [['conv3_maxpool'], ['fc1']]
            px_s = ProxyNet1D(num_classes=10, input_dim=84)

    elif args.database == 'NYU-Depth-V2':
        if args.Tmodel == 'FuseNet-I':
            px_t = ProxyNet_resoFixed(num_classes=41, in_channels=64, up_sample=(480, 640))
            if args.Smodel == 'FuseNet-RGBbranch':
                feat_names = [['CBR2_RGBD_DEC'], ['CBR2_RGB_DEC']]
                px_s = ProxyNet_resoFixed(num_classes=41, in_channels=64, up_sample=(480, 640))
            elif args.Smodel == 'FuseNet-Dbranch':
                feat_names = [['CBR2_RGBD_DEC'], ['CBR2_D_DEC']]
                px_s = ProxyNet_resoFixed(num_classes=41, in_channels=64, up_sample=(480, 640))
        elif args.Tmodel == 'FuseNet-RGBbranch':
            px_t = ProxyNet_resoFixed(num_classes=41, in_channels=64, up_sample=(480, 640))
            feat_names = [['CBR2_RGB_DEC'], ['CBR2_D_DEC']]
            px_s = ProxyNet_resoFixed(num_classes=41, in_channels=64, up_sample=(480, 640))
        elif args.Tmodel == 'FuseNet-Dbranch':
            px_t = ProxyNet_resoFixed(num_classes=41, in_channels=64, up_sample=(480, 640))
            feat_names = [['CBR2_D_DEC'], ['CBR2_RGB_DEC']]
            px_s = ProxyNet_resoFixed(num_classes=41, in_channels=64, up_sample=(480, 640))

    elif args.database == 'RAVDESS':
        if args.Tmodel == 'DSCNN-I':
            px_t = ProxyNet1D(num_classes=8, input_dim=320)
            if args.Smodel in ['AudioBranchNet', 'VisualBranchNet']:
                feat_names = [['fc1'], ['fc1']]
                px_s = ProxyNet1D(num_classes=8, input_dim=320)
        elif args.Tmodel in ['AudioBranchNet', 'VisualBranchNet']:
            px_t = ProxyNet1D(num_classes=8, input_dim=320)
            if args.Smodel in ['AudioBranchNet', 'VisualBranchNet']:
                feat_names = [['fc1'], ['fc1']]
                px_s = ProxyNet1D(num_classes=8, input_dim=320)

    elif args.database == 'VGGSound-50k':
        if args.Tmodel == 'DSCNN-VGGS-I':
            px_t = ProxyNet1D(num_classes=141, input_dim=320)
            if args.Smodel in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
                feat_names = [['fc1'], ['fc1']]
                px_s = ProxyNet1D(num_classes=141, input_dim=320)
        elif args.Tmodel in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
            px_t = ProxyNet1D(num_classes=141, input_dim=320)
            if args.Smodel in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
                feat_names = [['fc1'], ['fc1']]
                px_s = ProxyNet1D(num_classes=141, input_dim=320)

    elif args.database == 'CMMD-V2':
        if args.Tmodel == 'MLP-I':
            px_t = ProxyNet1D(num_classes=8, input_dim=256)
            if args.Smodel in ['MLP-Vb', 'MLP-Tb']:
                feat_names = [['fc2'], ['fc2']]
                px_s = ProxyNet1D(num_classes=8, input_dim=512)
        elif args.Tmodel in ['MLP-Vb', 'MLP-Tb']:
            px_t = ProxyNet1D(num_classes=8, input_dim=512)
            if args.Smodel in ['MLP-Vb', 'MLP-Tb']:
                feat_names = [['fc2'], ['fc2']]
                px_s = ProxyNet1D(num_classes=8, input_dim=512)

    else:
        raise ValueError(f"Invalid database name {args.database}.")

    KL_batchmean = torch.nn.KLDivLoss(reduction='batchmean')
    KL_none = torch.nn.KLDivLoss(reduction='none')
    return feat_names, px_t, px_s, KL_batchmean, KL_none


class ProxyNet(nn.Module):
    """Proxy network for C$^2$KD, serving as either a teacher or student model"""

    def __init__(self, num_classes=28, in_channels=256):
        super(ProxyNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = conv_1x1_bn(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def conv_1x1_bn(num_input_channels, num_mid_channel):
    return nn.Sequential(
        conv1x1(num_input_channels, num_mid_channel),
        nn.BatchNorm2d(num_mid_channel),
        nn.LeakyReLU(0.1, inplace=True),
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ProxyNet1D(nn.Module):
    """Proxy network for C$^2$KD, adapted to process 1D vector data with shape (batch_size, dim)"""

    def __init__(self, num_classes=28, input_dim=256):
        super(ProxyNet1D, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)  # Equivalent to conv1x1 layer
        self.bn = nn.BatchNorm1d(input_dim)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = nn.Linear(input_dim, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def conv_1x1_bn_resoFixed(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class ProxyNet_resoFixed(nn.Module):
    """Proxy network for C$^2$KD, serving as either a teacher or student model"""

    def __init__(self, num_classes=28, in_channels=256, up_sample=None):
        super(ProxyNet_resoFixed, self).__init__()
        self.layer = conv_1x1_bn_resoFixed(in_channels, in_channels)  # 保持特征图分辨率
        self.fc = nn.Conv2d(in_channels, num_classes, kernel_size=1)  # 改为1x1卷积替代全连接层
        self.up_sample = up_sample

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer(x)
        x = self.fc(x)  # 输出特征图分辨率不变
        x = F.interpolate(x, size=self.up_sample, mode='bilinear', align_corners=False) if self.up_sample else x
        return x


def ntkl(logits_student, logits_teacher, target, mask=None, criterion4=None, temperature=1):

    gt_mask = _get_gt_mask(logits_student, target)
    logits_teacher = logits_teacher * (~gt_mask)
    pred_teacher_part2 = F.softmax(logits_teacher / temperature, dim=1)
    logits_student = logits_student * (~gt_mask)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature, dim=1)
    if mask.sum() == 0:
        temp = torch.tensor(0)
    else:
        temp = ((mask * (criterion4(log_pred_student_part2, pred_teacher_part2.detach()).sum(1)))).mean()
    return temp


def ntkl_ss(logits_student, logits_teacher, target, original_shape, mask=None, criterion3=None, temperature=1):
    gt_mask = _get_gt_mask(logits_student, target)
    logits_teacher = logits_teacher * (~gt_mask)
    pred_teacher_part2 = F.softmax(logits_teacher / temperature, dim=1)
    logits_student = logits_student * (~gt_mask)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature, dim=1)

    b, h, w = original_shape
    log_pred_student_part2 = log_pred_student_part2.view(b, h * w, -1)
    pred_teacher_part2 = pred_teacher_part2.view(b, h * w, -1)

    if mask.sum() == 0:
        temp = torch.tensor(0)
    else:
        temp = criterion3(log_pred_student_part2, pred_teacher_part2.detach())/(h*w)
    return temp


def _get_gt_mask(logits, target):
    if target.dim() ==2:
        mask = target.bool()
        return mask
    else:
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask


def get_feats(args, features, feat_name):
    if args.database == 'NYU-Depth-V2' and args.Tmodel == 'CEN':
        if args.mode == 'm1':
            return features[feat_name][0]
        elif args.mode == 'm2':
            return features[feat_name][1]
    else:
        return features[feat_name]


def get_logits(args, outputs):
    if isinstance(outputs, tuple):
        if len(outputs) == 4:
            return outputs[1]
    elif outputs.dim() == 4 and args.database != 'NYU-Depth-V2':
        return outputs.view(-1)
    else:
        return outputs


def get_labels(args, label):
    if args.database == 'NYU-Depth-V2':
        # [bs, 480, 640]
        one_hot_label = F.one_hot(label, num_classes=41).permute(0, 3, 1, 2).float()
        return one_hot_label
    elif args.database == 'VGGSound-50k':
        return label[:, 0, -1].long()
    else:
        return label


if __name__ == '__main__':
    # input_tensor_1 = torch.randn(2, 256, 28, 14)
    # input_tensor_2 = torch.randn(2, 256)

    # Test the ProxyNet model
    model1 = ProxyNet(10, 128)
    # Test the ProxyNet1D model
    model2 = ProxyNet1D(10, 84)

    # res1 = model1(input_tensor_1)
    # res2 = model2(input_tensor_2)

    # print(res1.shape)
    # print(res2.shape)

    # summary(model1, input_size=(1, 128, 14, 14))
    # summary(model2, input_size=(1, 84))


