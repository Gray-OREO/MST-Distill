import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def get_RKDmodules(args, model_s, n_data):
    if args.database == 'AV-MNIST':
        if args.Tmodel == 'CNN-I':
            t_dim = 298
            if args.Smodel == 'LeNet5':
                feat_names = ['fc1', 'fc1']
                s_dim = 84
            elif args.Smodel == 'ThreeLayerCNN-A':
                feat_names = ['fc1', 'conv3']
                s_dim = 128 * 14 * 14
        elif args.Tmodel == 'LeNet5':
            t_dim = 84
            if args.Smodel == 'ThreeLayerCNN-A':
                feat_names = ['fc1', 'conv3']
                s_dim = 128 * 14 * 14
        elif args.Tmodel == 'ThreeLayerCNN-A':
            t_dim = 128 * 14 * 14
            if args.Smodel == 'LeNet5':
                feat_names = ['conv3', 'fc1']
                s_dim = 84

    elif args.database == 'RAVDESS':
        if args.Tmodel == 'DSCNN-I':
            t_dim = 160
            if args.Smodel in ['AudioBranchNet', 'VisualBranchNet']:
                feat_names = ['fc2', 'fc2']
                s_dim = 160
        elif args.Tmodel in ['AudioBranchNet', 'VisualBranchNet']:
            t_dim = 160
            if args.Smodel in ['AudioBranchNet', 'VisualBranchNet']:
                feat_names = ['fc2', 'fc2']
                s_dim = 160

    elif args.database == 'VGGSound-50k':
        if args.Tmodel == 'DSCNN-VGGS-I':
            t_dim = 160
            if args.Smodel in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
                feat_names = ['fc2', 'fc2']
                s_dim = 160
        elif args.Tmodel in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
            t_dim = 160
            if args.Smodel in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
                feat_names = ['fc2', 'fc2']
                s_dim = 160

    elif args.database == 'CMMD-V2':
        if args.Tmodel == 'MLP-I':
            t_dim = 256
            if args.Smodel in ['MLP-Vb', 'MLP-Tb']:
                feat_names = ['fc2', 'fc2']
                s_dim = 512
        elif args.Tmodel in ['MLP-Vb', 'MLP-Tb']:
            t_dim = 512
            if args.Smodel in ['MLP-Vb', 'MLP-Tb']:
                feat_names = ['fc2', 'fc2']
                s_dim = 512

    else:
        raise ValueError(f"Invalid database name {args.database}.")
    criterion = RKDLoss(s_dim, t_dim, 128).cuda(args.cuda_id)
    proj_s, proj_t = criterion.embed_s, criterion.embed_t
    proj_s.cuda(args.cuda_id)
    proj_t.cuda(args.cuda_id)
    params = list(model_s.parameters()) + list(proj_s.parameters()) + list(proj_t.parameters())
    optim = torch.optim.Adam(params, lr=args.lr)
    return feat_names, criterion, optim


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss


class RKDLoss(nn.Module):
    def __init__(self, s_dim, t_dim, feat_dim):
        super(RKDLoss, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        self.dist_loss = RkdDistance()
        self.dist_angle = RKdAngle()

    def forward(self, f_s, f_t):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        loss_d = self.dist_loss(f_s, f_t)
        loss_a = self.dist_angle(f_s, f_t)
        loss = 1*loss_d + 2*loss_a
        return loss


def penultimate_feature_extractor(feat_name, features, args):
    if args.database == 'AV-MNIST':
        penultimate_feature = features[feat_name]
        if penultimate_feature.dim() == 4:
            penultimate_feature = F.max_pool2d(penultimate_feature, 2).view(-1, 128 * 14 * 14)
    elif args.database == 'VGGSound-50k':
        if feat_name == 'psp':
            penultimate_feature = features[feat_name][0].mean(dim=1)
        else:
            penultimate_feature = features[feat_name]
    else:
        penultimate_feature = features[feat_name]
    return penultimate_feature

if __name__ == '__main__':
    emb1 = Embed(84, 128)
    emb2 = Embed(128 * 14 * 14, 128)

    # summary(emb1, input_size=(1, 84))
    summary(emb2, input_size=(1, 128 * 14 * 14))
