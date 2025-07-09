from copy import deepcopy
import torch
import torch.nn as nn


def my_permute_new(x, index):
    if isinstance(x, list):
        y = deepcopy(x)
        perm_index = torch.randperm(x[0].shape[0])
        for i in index:
            y[0][:, i] = x[0][perm_index, i]
            y[1][:, i] = x[1][perm_index, i]
        return y
    else:
        y = x.clone()
        perm_index = torch.randperm(x.shape[0])
        for i in index:
            y[:, i] = x[perm_index, i]
        return y


def my_freeze_new(x, index):  # in-place modification
    if isinstance(x, list):
        y = deepcopy(x)
        # 计算每个列表元素在指定索引上的均值
        tmp_mean_0 = x[0][:, index].mean(dim=0)
        tmp_mean_1 = x[1][:, index].mean(dim=0)
        # 将列表中的每个元素在指定索引上修改为均值
        y[0][:, index] = tmp_mean_0
        y[1][:, index] = tmp_mean_1
        return y
    else:
        y = x.clone()
        # 计算指定索引的均值
        tmp_mean = x[:, index].mean(dim=0)
        # 将指定索引上的值设置为均值
        y[:, index] = tmp_mean
        return y


def my_change(x, change_type, index):
    if change_type == 'permute':
        return my_permute_new(x, index)
    elif change_type == 'freeze':
        return my_freeze_new(x, index)
    else:
        raise ValueError("Undefined change_type")


def hook_fn(name, features, fn, axis):
    def hook(module, input, output):
        if fn == 'zero':
            res = torch.zeros_like(output)
        elif fn == 'permute':
            res = my_change(output, 'permute', axis)
        else:
            res = output
        features[name] = res
        return res
    return hook


def hooks_builder(model, hook_names, fn=None, axis=None):
    features = {}
    hooks = []
    for name in hook_names:
        submodule = get_submodule(model, name)
        hook = submodule.register_forward_hook(hook_fn(name, features, fn, axis))
        hooks.append(hook)
    return hooks, features


def get_submodule(model, submodule_name):
    """递归获取子模块"""
    names = submodule_name.split('.')
    submodule = model
    for name in names:
        submodule = submodule._modules[name]
    return submodule


def hooks_remover(hooks):
    for hook in hooks:
        hook.remove()


def get_MGDFRmodules(args):
    feat_name, feat_dim = None, None
    if args.database == 'AV-MNIST':
        if args.Tmodel == 'CNN-I':
            feat_name = ['fc1']
            feat_dim = 298  # [bs, 20]
        elif args.Tmodel == 'LeNet5':
            feat_name = ['fc1']
            feat_dim = 84  # [bs, 84]
        elif args.Tmodel == 'ThreeLayerCNN-A':
            feat_name = ['conv3']
            feat_dim = 128  # [bs, 128, 28, 28]

    elif args.database == 'NYU-Depth-V2':
        if args.Tmodel == 'FuseNet-I':
            feat_name = ['after_fusion_identity']
            feat_dim = 512  # [bs, 512, 15, 20]
        elif args.Tmodel == 'FuseNet-RGBbranch':
            feat_name = ['CBR5_RGB_ENC']  # [bs, 512, 30, 40]
            feat_dim = 512
        elif args.Tmodel == 'FuseNet-Dbranch':
            feat_name = ['CBR5_DEPTH_ENC']  # [bs, 512, 30, 40]
            feat_dim = 512


    elif args.database == 'RAVDESS':
        if args.Tmodel == 'DSCNN-I':
            feat_name = ['fc2']
            feat_dim = 160  # [bs, 160]
        elif args.Tmodel in ['VisualBranchNet', 'AudioBranchNet']:
            feat_name = ['fc2']
            feat_dim = 160  # [bs, 160]

    elif args.database == 'VGGSound-50k':
        if args.Tmodel == 'DSCNN-VGGS-I':
            feat_name = ['fc2']
            feat_dim = 160  # [bs, 160]
        elif args.Tmodel in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
            feat_name = ['fc2']
            feat_dim = 160  # [bs, 160]

    elif args.database == 'CMMD-V2':
        if args.Tmodel == 'MLP-I':
            feat_name = ['fc1']
            feat_dim = 512  # [bs, 20]
        elif args.Tmodel in ['MLP-Vb', 'MLP-Tb']:
            feat_name = ['fc2']
            feat_dim = 512  # [bs, 84]

    else:
        raise ValueError("Undefined database")
    criterion = DistLoss(args)
    salience_vector = torch.zeros(args.repeat_permute, feat_dim)
    return feat_name, feat_dim, salience_vector, criterion


def get_feat(database, outputs):
    if database == 'NYU-Depth-V2':
        res = outputs[0][-1]
    elif database == 'VGGSound-50k':
        res = outputs[1] if isinstance(outputs, tuple) else outputs
    else:
        res = outputs
    return res



class DistLoss(nn.Module):
    def __init__(self, args):
        super(DistLoss, self).__init__()
        self.database_name = args.database
        self.loss = nn.MSELoss()

    def forward(self, out_t, out_s):
        student_logits = get_feat(self.database_name, out_s)
        teacher_logits = get_feat(self.database_name, out_t)
        return self.loss(teacher_logits, student_logits)
