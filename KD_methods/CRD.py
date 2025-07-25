import torch
import torch.nn as nn
import torch.nn.functional as F
import math

eps = 1e-7


def get_CRDmodules(args, model_s, n_data):
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
    criterion = CRDLoss(s_dim, t_dim, 128, 4096, args.nce_t, 0.5, n_data, args).cuda(args.cuda_id)
    proj_s, proj_t = criterion.embed_s, criterion.embed_t
    proj_s.cuda(args.cuda_id)
    proj_t.cuda(args.cuda_id)
    params = list(model_s.parameters()) + list(proj_s.parameters()) + list(proj_t.parameters())
    optim = torch.optim.Adam(params, lr=args.lr)
    return feat_names, criterion, optim


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, s_dim, t_dim, feat_dim, nce_k, nce_t, nce_m, n_data, args):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        self.contrast = ContrastMemory(feat_dim, n_data, nce_k, nce_t, nce_m, args.cuda_id)
        self.criterion_t = ContrastLoss(n_data)
        self.criterion_s = ContrastLoss(n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """

        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
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


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


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


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, cuda_id=0):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams, cuda_id)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            # print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            # print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs, cuda_id):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)
        self.cuda_id = cuda_id

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda(self.cuda_id)
        self.alias = self.alias.cuda(self.cuda_id)

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj