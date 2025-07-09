import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


def get_OFAmodules(args, model_s):
    feat_names = [[], []]
    projectors = []
    if args.database == 'AV-MNIST':
        cls_num = 10
        if args.Tmodel == 'CNN-I':
            if args.Smodel == 'LeNet5':
                feat_names = [['fc2'], ['conv1', 'conv2', 'fc1']]
                proj_1 = Projector(6, 2*cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector(16, 2*cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector_1D(84, 2*cls_num, cls_num)
                projectors.append(proj_3)
            elif args.Smodel == 'ThreeLayerCNN-A':
                feat_names = [['fc2'], ['conv1', 'conv2', 'conv3']]
                proj_1 = Projector(32, 2*cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector(64, 2*cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector(128, 2*cls_num, cls_num)
                projectors.append(proj_3)
        elif args.Tmodel == 'LeNet5':
            if args.Smodel == 'ThreeLayerCNN-A':
                feat_names = [['fc2'], ['conv1', 'conv2', 'conv3']]
                proj_1 = Projector(32, 2 * cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector(64, 2 * cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector(128, 2 * cls_num, cls_num)
                projectors.append(proj_3)
        elif args.Tmodel == 'ThreeLayerCNN-A':
            if args.Smodel == 'LeNet5':
                feat_names = [['fc1'], ['conv1', 'conv2', 'fc1']]
                proj_1 = Projector(6, 2 * cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector(16, 2 * cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector_1D(84, 2 * cls_num, cls_num)
                projectors.append(proj_3)

    elif args.database == 'RAVDESS':
        cls_num = 8
        if args.Tmodel == 'DSCNN-I':
            if args.Smodel == 'AudioBranchNet':
                feat_names = [['fc3'], ['audio_branch', 'fc1', 'fc2']]
                proj_1 = Projector_1D(1280, 2*cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(320, 2*cls_num, cls_num)
                projectors.append(proj_2 )
                proj_3 = Projector_1D(160, 2*cls_num, cls_num)
                projectors.append(proj_3)
            elif args.Smodel == 'VisualBranchNet':
                feat_names = [['fc3'], ['visual_branch', 'fc1', 'fc2']]
                proj_1 = Projector_1D(18816, 2*cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(320, 2*cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector_1D(160, 2*cls_num, cls_num)
                projectors.append(proj_3)
        elif args.Tmodel == 'VisualBranchNet':
            if args.Smodel == 'AudioBranchNet':
                feat_names = [['fc3'], ['audio_branch', 'fc1', 'fc2']]
                proj_1 = Projector_1D(1280, 2 * cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(320, 2 * cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector_1D(160, 2 * cls_num, cls_num)
                projectors.append(proj_3)
        elif args.Tmodel == 'AudioBranchNet':
            if args.Smodel == 'VisualBranchNet':
                feat_names = [['fc3'], ['visual_branch', 'fc1', 'fc2']]
                proj_1 = Projector_1D(18816, 2 * cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(320, 2 * cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector_1D(160, 2 * cls_num, cls_num)
                projectors.append(proj_3)

    elif args.database == 'VGGSound-50k':
        cls_num = 141
        if args.Tmodel == 'DSCNN-VGGS-I':
            if args.Smodel == 'VisualBranchNet-VGGS':
                feat_names = [['fc3'], ['visual_branch', 'fc1', 'fc2']]
                proj_1 = Projector_1D(128, 2*cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(320, 2*cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector_1D(160, 2*cls_num, cls_num)
                projectors.append(proj_3)
            elif args.Smodel == 'AudioBranchNet-VGGS':
                feat_names = [['fc3'], ['audio_branch', 'fc1', 'fc2']]
                proj_1 = Projector_1D(256, 2*cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(320, 2*cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector_1D(160, 2*cls_num, cls_num)
                projectors.append(proj_3)
        elif args.Tmodel == 'VisualBranchNet-VGGS':
            if args.Smodel == 'AudioBranchNet-VGGS':
                feat_names = [['fc3'], ['audio_branch', 'fc1', 'fc2']]
                proj_1 = Projector_1D(256, 2 * cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(320, 2 * cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector_1D(160, 2 * cls_num, cls_num)
                projectors.append(proj_3)
        elif args.Tmodel == 'AudioBranchNet-VGGS':
            if args.Smodel == 'VisualBranchNet-VGGS':
                feat_names = [['fc3'], ['visual_branch', 'fc1', 'fc2']]
                proj_1 = Projector_1D(128, 2 * cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(320, 2 * cls_num, cls_num)
                projectors.append(proj_2)
                proj_3 = Projector_1D(160, 2 * cls_num, cls_num)
                projectors.append(proj_3)

    elif args.database == 'CMMD-V2':
        cls_num = 8
        if args.Tmodel == 'MLP-I':
            if args.Smodel in ['MLP-Vb', 'MLP-Tb']:
                feat_names = [['fc3'], ['fc1', 'fc2']]
                proj_1 = Projector_1D(1024, 2 * cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(512, 2 * cls_num, cls_num)
                projectors.append(proj_2)

        elif args.Tmodel in ['MLP-Vb', 'MLP-Tb']:
            if args.Smodel in ['MLP-Vb', 'MLP-Tb']:
                feat_names = [['fc3'], ['fc1', 'fc2']]
                proj_1 = Projector_1D(1024, 2 * cls_num, cls_num)
                projectors.append(proj_1)
                proj_2 = Projector_1D(512, 2 * cls_num, cls_num)
                projectors.append(proj_2)

    else:
        raise ValueError(f"Invalid database name {args.database}.")
    criterion = OFA_Loss(feat_names, args, projectors, cls_num)
    params = list(model_s.parameters())
    for proj in projectors:
        proj.apply(init_weights)
        params += list(proj.parameters())
    optim = torch.optim.Adam(params, lr=args.lr)
    return projectors, criterion, optim


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class Projector(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(Projector, self).__init__()
        down_sample_blks = [SepConv(in_channels, 2*in_channels),
                            SepConv(2*in_channels, out_channels)]
        self.blks = nn.Sequential(
            *down_sample_blks,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes)
        )

    def forward(self, x):
        return self.blks(x)


class Projector_1D(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(Projector_1D, self).__init__()
        self.blks = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.blks(x)


def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class OFA_Loss(nn.Module):
    def __init__(self, feat_names, args, projectors, cls_num):
        super(OFA_Loss, self).__init__()
        self.feat_names = feat_names
        self.database = args.database
        self.mode = args.mode
        self.eps = args.ofa_eps
        self.temperature = args.ofa_temperature
        self.projectors = projectors
        self.cls_num = cls_num

    def forward(self, teacher_feats, student_feats, labels):
        loss = 0
        tea_logits = teacher_feats[self.feat_names[0][0]].detach()
        N = len(self.feat_names[1])
        labels = labels[:, 0, -1].long() if self.database == 'VGGSound-50k' else labels
        target_mask = F.one_hot(labels, self.cls_num)
        for i in range(N):
            stu_feat = student_feats[self.feat_names[1][i]]
            proj = self.projectors[i]
            stu_logits = proj(stu_feat)
            loss += ofa_loss(stu_logits, tea_logits, target_mask, self.eps, self.temperature)
        return loss / N


def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()


def projectors_train(projectors):
    for projector in projectors:
        projector.train()


def projectors_eval(projectors):
    for projector in projectors:
        projector.eval()


if __name__ == '__main__':
    input_tensor = torch.randn(3, 84)
    proj = Projector_1D(84, 20, 10)
    output = proj(input_tensor)
    print(output.shape)