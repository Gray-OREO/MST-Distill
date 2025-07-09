import torch.nn.functional as F
import torch.nn as nn
import torch
from torchinfo import summary


def get_MSTmodules(args):
    T_feat_names, A_feat_names, S_feat_names = None, None, None
    T_feat_shapes, A_feat_shapes, S_feat_shapes = None, None, None

    assert args.k <= (2 * args.mask_layer_num), f"Invalid Setting in k '{args.k} or mask_layer_num {args.mask_layer_num}'!"

    if args.database == 'AV-MNIST':
        cls_num = 10
        if args.Tmodel == 'CNN-I':
            T_feat_names = ['concat_identity', 'fc1']
            T_feat_shapes = [(596,), (298,)] # [bs, 596], [bs, 298]
        else:
            raise ValueError(f"Invalid Setting in Tmodel '{args.Tmodel}'!")

        if args.Smodel == 'LeNet5':
            A_feat_names = ['conv2', 'conv3']
            A_feat_shapes = [(64, 56, 56), (128, 28, 28)]
        elif args.Smodel == 'ThreeLayerCNN-A':
            A_feat_names = ['conv3', 'fc1']
            A_feat_shapes = [(120, 1, 1), (84,)]
        else:
            raise ValueError(f"Invalid Setting in Smodel '{args.Smodel}'!")


    elif args.database == 'NYU-Depth-V2':
        cls_num = 41
        if args.Tmodel == 'FuseNet-I':
            T_feat_names = ['after_fusion_identity', 'CBR5_RGBD_DEC']
            T_feat_shapes = [(512, 15, 20), (512, 30, 40)]
        else:
            raise ValueError(f"Invalid Setting in Tmodel '{args.Tmodel}'!")

        if args.Smodel == 'FuseNet-RGBbranch':
            A_feat_names = ['CBR5_DEPTH_ENC', 'CBR5_D_DEC']
            A_feat_shapes = [(512, 30, 40), (512, 30, 40)]
        elif args.Smodel == 'FuseNet-Dbranch':
            A_feat_names = ['CBR5_RGB_ENC', 'CBR5_RGB_DEC']
            A_feat_shapes = [(512, 30, 40), (512, 30, 40)]
        else:
            raise ValueError(f"Invalid Setting in Smodel '{args.Smodel}'!")


    elif args.database == 'RAVDESS':
        cls_num = 8
        if args.Tmodel == 'DSCNN-I':
            T_feat_names = ['fc1', 'fc2']
            T_feat_shapes = [(320,), (160,)]
        else:
            raise ValueError(f"Invalid Setting in Tmodel '{args.Tmodel}'!")

        if args.Smodel in ['VisualBranchNet', 'AudioBranchNet']:
            A_feat_names = ['fc1', 'fc2']
            A_feat_shapes = [(320,), (160,)]
        else:
            raise ValueError(f"Invalid Setting in Smodel '{args.Smodel}'!")


    elif args.database == 'VGGSound-50k':
        cls_num = 141
        if args.Tmodel == 'DSCNN-VGGS-I':
            T_feat_names = ['fc1', 'fc2']
            T_feat_shapes = [(320,), (160,)]
        else:
            raise ValueError(f"Invalid Setting in Tmodel '{args.Tmodel}'!")

        if args.Smodel in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
            A_feat_names = ['fc1', 'fc2']
            A_feat_shapes = [(320,), (160,)]
        else:
            raise ValueError(f"Invalid Setting in Smodel '{args.Smodel}'!")


    elif args.database == 'CMMD-V2':
        cls_num = 8
        if args.Tmodel == 'MLP-I':
            T_feat_names = ['fc1', 'fc2']
            T_feat_shapes = [(512,), (256,)]
        else:
            raise ValueError(f"Invalid Setting in Tmodel '{args.Tmodel}'!")

        if args.Smodel in ['MLP-Vb', 'MLP-Tb']:
            A_feat_names = ['fc1', 'fc2']
            A_feat_shapes = [(1024,), (512,)]
        else:
            raise ValueError(f"Invalid Setting in Smodel '{args.Smodel}'!")

    else:
        raise ValueError(f'Invalid Setting in database "{args.database}"!')

    MaskNets_T, MaskNets_A = [], []

    for i, (feat_name, feat_shape) in enumerate(zip(T_feat_names, T_feat_shapes)):
        if len(feat_shape) == 1:
            feat_dim = feat_shape[0]
            MaskNets_T.append(MaskNet_1d(feat_dim, hidden_dim=feat_dim//(2*args.mask_head_num)*args.mask_head_num, num_heads=args.mask_head_num))
        else:
            feat_dim = feat_shape[0]
            if feat_shape[-1] == 1:
                MaskNets_T.append(MaskNet_1d(feat_dim, hidden_dim=feat_dim//(2*args.mask_head_num)*args.mask_head_num, num_heads=args.mask_head_num))
            else:
                MaskNets_T.append(MaskNet_2d(feat_dim, hidden_dim=feat_dim//(2*args.mask_head_num)*args.mask_head_num, num_heads=args.mask_head_num))

    for i, (feat_name, feat_shape) in enumerate(zip(A_feat_names, A_feat_shapes)):
        if len(feat_shape) == 1:
            feat_dim = feat_shape[0]
            MaskNets_A.append(MaskNet_1d(feat_dim, hidden_dim=feat_dim//(2*args.mask_head_num)*args.mask_head_num, num_heads=args.mask_head_num))
        else:
            feat_dim = feat_shape[0]
            if feat_shape[-1] == 1:
                MaskNets_A.append(MaskNet_1d(feat_dim, hidden_dim=feat_dim//(2*args.mask_head_num)*args.mask_head_num, num_heads=args.mask_head_num))
            else:
                MaskNets_A.append(MaskNet_2d(feat_dim, hidden_dim=feat_dim//(2*args.mask_head_num)*args.mask_head_num, num_heads=args.mask_head_num))

    # Loading mask_layer_num for DTD
    assert args.mask_layer_num <= len(T_feat_names) and args.mask_layer_num <= len(A_feat_names), \
        f"Invalid Setting in mask_layer_num '{args.mask_layer_num}'!"
    T_feat_names = T_feat_names[:args.mask_layer_num]
    A_feat_names = A_feat_names[:args.mask_layer_num]
    MaskNets_T = MaskNets_T[:args.mask_layer_num]
    MaskNets_A = MaskNets_A[:args.mask_layer_num]

    ways = len(T_feat_names) + len(A_feat_names)
    gn = GateNet(cls_num, ways) if args.database != 'NYU-Depth-V2' else GateNet_dense(cls_num, ways)
    return T_feat_names, A_feat_names, MaskNets_T, MaskNets_A, gn


class MaskNet_1d(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads=8):
        super(MaskNet_1d, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.seq_len = feat_dim
        self.embed = nn.Linear(feat_dim, hidden_dim * feat_dim)
        self.mha = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 4 and x.shape[-1] == 1 and x.shape[-2] == 1:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x_embed = self.embed(x)  # (feat_dim, hidden_dim * feat_dim)
        x = x_embed.reshape(x.size(0), self.seq_len, self.hidden_dim)  # (batch_size, feat_dim, hidden_dim)
        # (batch_size, feat_dim, hidden_dim), (batch_size, num_heads, feat_dim, feat_dim)
        attn_output, attn_weights = self.mha(x, x, x)
        mask = self.fc(attn_output)  # (batch_size, feat_dim, 1)
        mask = mask.squeeze(-1)  # (batch_size, feat_dim)
        mask = self.sigmoid(mask)
        return mask


class SpatialReduction(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=feat_dim,
                out_channels=feat_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=feat_dim,
                bias=False
            ),

            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """input: (B, C, H, W) → output: (B, C)"""
        x = self.downsample(x)  # H/2, W/2
        x = self.gap(x)  # H=1, W=1
        return x.squeeze(-1).squeeze(-1)  # (B, C)


class MaskNet_2d(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.seq_len = feat_dim
        self.hidden_dim = hidden_dim
        self.spatial_reduce = SpatialReduction(feat_dim)
        self.embed = nn.Linear(feat_dim, hidden_dim * feat_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.spatial_reduce(x)  # (batch_size, feat_dim)
        x_embed = self.embed(x)  # (feat_dim, hidden_dim * feat_dim)
        x = x_embed.reshape(x.size(0), self.seq_len, self.hidden_dim)  # (batch_size, feat_dim, hidden_dim)
        # (batch_size, feat_dim, hidden_dim), (batch_size, num_heads, feat_dim, feat_dim)
        attn_output, attn_weights = self.mha(x, x, x)
        mask = self.fc(attn_output)  # (batch_size, feat_dim, 1)
        mask = mask.squeeze(-1)  # (batch_size, feat_dim)
        mask = self.sigmoid(mask)
        return mask


def hook_fn(name, features, masknet):
    def hook(module, input, output):
        if len(output.shape) == 4 and output.shape[-1] == 1 and output.shape[-2] == 1:
            output = output.squeeze(-1).squeeze(-1)
        mask = masknet(output)
        if len(mask.shape) != len(output.shape):
            for _ in range(len(output.shape) - len(mask.shape)):
                mask = mask.unsqueeze(-1)
        res = mask * output
        features[name+'_bfMask'] = output
        features['Mask'] = mask
        features[name] = res
        return res
    return hook


def hook_builder(model, hook_name, masknet):
    features = {}
    submodule = get_submodule(model, hook_name)
    hook = submodule.register_forward_hook(hook_fn(hook_name, features, masknet))
    return hook, features


def get_submodule(model, submodule_name):
    names = submodule_name.split('.')
    submodule = model
    for name in names:
        submodule = submodule._modules[name]
    return submodule


def distillation_loss(args, student_outputs, teacher_outputs, T=4.0):
    if args.database == 'NYU-Depth-V2':
        pred = student_outputs
        tar = teacher_outputs
        student_probs_T = F.log_softmax(pred / T, dim=1)
        teacher_probs_T_0 = F.softmax(tar / T, dim=1)
        kl_loss = F.kl_div(student_probs_T, teacher_probs_T_0, reduction='batchmean') * (T * T) * 1/(pred.shape[2] * pred.shape[3])
    elif args.database == 'VGGSound-50k' and args.Tmodel == 'CPSP':
        teacher_outputs = teacher_outputs[1]
        student_probs_T = F.log_softmax(student_outputs / T, dim=1)
        teacher_probs_T = F.softmax(teacher_outputs / T, dim=1)
        kl_loss = F.kl_div(student_probs_T, teacher_probs_T, reduction='batchmean') * (T * T)
    else:
        student_probs_T = F.log_softmax(student_outputs / T, dim=1)
        teacher_probs_T = F.softmax(teacher_outputs / T, dim=1)
        kl_loss = F.kl_div(student_probs_T, teacher_probs_T, reduction='batchmean') * (T * T)
    return kl_loss


def distillation_loss_detached(args, student_outputs, teacher_outputs, T=4.0):
    if args.database == 'NYU-Depth-V2':
        pred = student_outputs
        tar = teacher_outputs.detach()
        student_probs_T = F.log_softmax(pred / T, dim=1)
        teacher_probs_T_0 = F.softmax(tar / T, dim=1)
        kl_loss = F.kl_div(student_probs_T, teacher_probs_T_0, reduction='batchmean') * (T * T) * 1/(pred.shape[2] * pred.shape[3])
    elif args.database == 'VGGSound-50k' and args.Tmodel == 'CPSP':
        teacher_outputs = teacher_outputs[1].detach()
        student_probs_T = F.log_softmax(student_outputs / T, dim=1)
        teacher_probs_T = F.softmax(teacher_outputs / T, dim=1)
        kl_loss = F.kl_div(student_probs_T, teacher_probs_T, reduction='batchmean') * (T * T)
    else:
        teacher_outputs = teacher_outputs.detach()
        student_probs_T = F.log_softmax(student_outputs / T, dim=1)
        teacher_probs_T = F.softmax(teacher_outputs / T, dim=1)
        kl_loss = F.kl_div(student_probs_T, teacher_probs_T, reduction='batchmean') * (T * T)
    return kl_loss


class GateNet(nn.Module):
    def __init__(self, cls_num, ways):
        super(GateNet, self).__init__()

        self.fc1 = nn.Linear(cls_num, 2 * cls_num)
        self.fc2 = nn.Linear(2 * cls_num, ways)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class GateNet_dense(nn.Module):
    def __init__(self, cls_num, ways):
        super(GateNet_dense, self).__init__()
        hidden_dim = 2 * cls_num

        self.conv_point = nn.Conv2d(in_channels=cls_num,
                                    out_channels=hidden_dim,
                                    kernel_size=1,
                                    stride=1)
        self.conv_depthwise = nn.Conv2d(in_channels=hidden_dim,
                                        out_channels=hidden_dim,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=hidden_dim)
        self.fc = nn.Linear(hidden_dim, ways)

    def forward(self, x):
        x = self.conv_point(x)  # (batch_size, hidden_dim, height, width)
        x = F.relu(x)
        x = self.conv_depthwise(x)  # (batch_size, hidden_dim, height/2, width/2)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # (batch_size, hidden_dim)
        x = self.fc(x)  # (batch_size, 2)
        return x


def TGroup_forward(x_ori, x1_ori, model_t, MaskNets_T, T_feat_names, preprocessing_t):
    outputs_t = []
    hooks_t = []
    for i, (MaskNet, feat_name) in enumerate(zip(MaskNets_T, T_feat_names)):
        x, x1 = x_ori.clone(), x1_ori.clone()
        hook_t, features_t = hook_builder(model_t, feat_name, MaskNet)
        output_t = model_t(x, x1) if preprocessing_t is None else preprocessing_t(model_t, x, x1)
        outputs_t.append(output_t)
        hooks_t.append(hook_t)
    return outputs_t, hooks_t


def AGroup_forward(x_ori, model_a, MaskNets_A, A_feat_names, preprocessing_a):
    outputs_a = []
    hooks_a = []
    for i, (MaskNet, feat_name) in enumerate(zip(MaskNets_A, A_feat_names)):
        x = x_ori.clone()
        hook_a, features_a = hook_builder(model_a, feat_name, MaskNet)
        output_a = model_a(x) if preprocessing_a is None else preprocessing_a(model_a, x)
        outputs_a.append(output_a)
        hooks_a.append(hook_a)
    return outputs_a, hooks_a


def TGroup_forward_TG(x_ori, x1_ori, T_Group, MaskNets_T, T_feat_names, preprocessing_t):
    outputs_t = []
    hooks_t = []
    for i, (MaskNet, model_t, feat_name) in enumerate(zip(MaskNets_T, T_Group, T_feat_names)):
        x, x1 = x_ori.clone(), x1_ori.clone()
        hook_t, features_t = hook_builder(model_t, feat_name, MaskNet)
        output_t = model_t(x, x1) if preprocessing_t is None else preprocessing_t(model_t, x, x1)
        outputs_t.append(output_t)
        hooks_t.append(hook_t)
    return outputs_t, hooks_t


def AGroup_forward_AG(x_ori, A_Group, MaskNets_A, A_feat_names, preprocessing_a):
    outputs_a = []
    hooks_a = []
    for i, (MaskNet, model_a, feat_name) in enumerate(zip(MaskNets_A, A_Group, A_feat_names)):
        x = x_ori.clone()
        hook_a, features_a = hook_builder(model_a, feat_name, MaskNet)
        output_a = model_a(x) if preprocessing_a is None else preprocessing_a(model_a, x)
        outputs_a.append(output_a)
        hooks_a.append(hook_a)
    return outputs_a, hooks_a


def load_balancing_loss(gate_probs, mode='KL'):
    if mode == 'KL':
        expert_avg_probs = gate_probs.mean(dim=0)  # shape: [num_experts]
        uniform_dist = torch.ones_like(expert_avg_probs) / expert_avg_probs.size(0)
        kl_loss = nn.KLDivLoss(reduction="batchmean")(
            torch.log(expert_avg_probs),
            uniform_dist
        )
        return kl_loss
    elif mode == 'CV':
        importance = gate_probs.sum(dim=0)  # shape: [num_experts]
        importance_mean = importance.mean()
        importance_std = importance.std()
        coeff_vairation = importance_std / (importance_mean)
        return coeff_vairation

    else:
        raise ValueError('Invalid Load Balancing Mode!')


def k_ways_MTGdistillation(args, outputs_MTG, outputs_s, gate_logits_softmax, k):
    if args.database == 'NYU-Depth-V2':
        stacked_MTG = torch.stack(outputs_MTG, dim=0).permute(1, 0, 2, 3, 4)
        _, topk_indices = torch.topk(gate_logits_softmax, k=k, dim=1)
        expanded_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 添加三个新维度
        expanded_indices = expanded_indices.expand(-1, -1, stacked_MTG.size(2), stacked_MTG.size(3),
                                                   stacked_MTG.size(4))
        selected_MTG = torch.gather(stacked_MTG, dim=1, index=expanded_indices)
        outputs_s_expanded = outputs_s.unsqueeze(1).expand(-1, k, -1, -1, -1)

    else:
        stacked_MTG = torch.stack(outputs_MTG, dim=0).permute(1, 0, 2)
        _, topk_indices = torch.topk(gate_logits_softmax, k=k, dim=1)
        selected_MTG = torch.gather(stacked_MTG, dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, stacked_MTG.size(2)))
        outputs_s_expanded = outputs_s.unsqueeze(1).expand(-1, k, -1)

    loss = 0
    for i in range(k):
        loss += distillation_loss(args, outputs_s_expanded[:, i], selected_MTG[:, i], T=4.0)
    return loss/k, topk_indices


def k_ways_MTGdistillation_detached(args, outputs_MTG, outputs_s, gate_logits_softmax, k):
    if args.database == 'NYU-Depth-V2':
        stacked_MTG = torch.stack(outputs_MTG, dim=0).permute(1, 0, 2, 3, 4)
        _, topk_indices = torch.topk(gate_logits_softmax, k=k, dim=1)
        expanded_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 添加三个新维度
        expanded_indices = expanded_indices.expand(-1, -1, stacked_MTG.size(2), stacked_MTG.size(3),
                                                   stacked_MTG.size(4))
        selected_MTG = torch.gather(stacked_MTG, dim=1, index=expanded_indices)
        outputs_s_expanded = outputs_s.unsqueeze(1).expand(-1, k, -1, -1, -1)

    else:
        stacked_MTG = torch.stack(outputs_MTG, dim=0).permute(1, 0, 2)
        _, topk_indices = torch.topk(gate_logits_softmax, k=k, dim=1)
        selected_MTG = torch.gather(stacked_MTG, dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, stacked_MTG.size(2)))
        outputs_s_expanded = outputs_s.unsqueeze(1).expand(-1, k, -1)

    loss = 0
    for i in range(k):
        loss += distillation_loss_detached(args, outputs_s_expanded[:, i], selected_MTG[:, i], T=4.0)
    return loss/k, topk_indices


def weighted_MTGdistillation_detached(args, outputs_MTG, outputs_s, gate_logits_softmax):
    if args.database == 'NYU-Depth-V2':
        # (num_teachers, batch_size, cls_num, h, w) -> (batch_size, num_teachers, cls_num, h, w)
        stacked_MTG = torch.stack(outputs_MTG, dim=0).permute(1, 0, 2, 3, 4)
        batch_size, num_teachers, cls_num, h, w = stacked_MTG.shape

        # (batch_size, num_teachers) -> (batch_size, num_teachers, 1, 1, 1)
        weights_expanded = gate_logits_softmax.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weights_expanded = weights_expanded.expand(-1, -1, cls_num, h, w)

        # (batch_size, num_teachers, cls_num, h, w)
        weighted_MTG = stacked_MTG * weights_expanded

        # (batch_size, cls_num, h, w)
        aggregated_MTG = torch.sum(weighted_MTG, dim=1)

    else:
        # (num_teachers, batch_size, cls_num) -> (batch_size, num_teachers, cls_num)
        stacked_MTG = torch.stack(outputs_MTG, dim=0).permute(1, 0, 2)
        batch_size, num_teachers, cls_num = stacked_MTG.shape
        # (batch_size, num_teachers) -> (batch_size, num_teachers, 1)
        weights_expanded = gate_logits_softmax.unsqueeze(-1)
        weights_expanded = weights_expanded.expand(-1, -1, cls_num)
        # (batch_size, num_teachers, cls_num)
        weighted_MTG = stacked_MTG * weights_expanded
        # (batch_size, cls_num)
        aggregated_MTG = torch.sum(weighted_MTG, dim=1)
    loss = distillation_loss_detached(args, outputs_s, aggregated_MTG, T=4.0)
    return loss, gate_logits_softmax


def hooks_remover(hooks):
    for hook in hooks:
        hook.remove()


if __name__ == '__main__':
    # 596 298 120 84
    m1 = MaskNet_1d(596, 596//(2*3)*3, num_heads=3)
    m2 = MaskNet_1d(298, 298//(2*3)*3, num_heads=3)
    m3 = MaskNet_1d(120, 120//(2*3)*3, num_heads=3)
    m4 = MaskNet_1d(84, 84//(2*3)*3, num_heads=3)

    summary(m1, input_size=(1,596))
    # summary(m2, input_size=(1,298))
    # summary(m3, input_size=(1,120))
    # summary(m4, input_size=(1,84))