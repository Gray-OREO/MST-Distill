import os, time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
import h5py
import platform
from models.CNNs import ThreeLayerCNN_A, IntermediateFusionNet
from models.LeNet5 import LeNet5
from models.FuseNet import define_FuseNet, define_branches
from models.SeqNets import DualStreamCNN, AudioBranchNet, VisualBranchNet
from models.CPSP import AudioBranchNet_VGGS, VisualBranchNet_VGGS, DualStreamCNN_VGGS
from models.MLPs import visualMLP, textualMLP, IntermediateFusionMLP
import torch.optim as optim
from functools import wraps
from tqdm import tqdm
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__} took {elapsed_time:.4f} seconds to run.")
        return result
    return wrapper


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_data(data_name):
    root = 'G:/Gray/Database_CMKD/'

    if data_name == 'AV-MNIST':
        return get_av_mnist(root)
    elif data_name == 'NYU-Depth-V2':
        return get_nyu_depth_v2(root)
    elif data_name == 'RAVDESS':
        return get_ravdess(root)
    elif data_name == 'VGGSound-50k':
        return get_vggsound(root)
    elif data_name == 'CMMD-V2':
        return get_cmmd_v2(root)
    else:
        raise ValueError(f'Invalid data name: {data_name}')


def get_av_mnist(root):
    train_data1 = np.load(root + 'avmnist/image/train_data.npy')
    train_data2 = np.load(root + 'avmnist/audio/train_data.npy')
    train_labels = np.load(root + 'avmnist/train_labels.npy')
    test_data1 = np.load(root + 'avmnist/image/test_data.npy')
    test_data2 = np.load(root + 'avmnist/audio/test_data.npy')
    test_labels = np.load(root + 'avmnist/test_labels.npy')
    data1 = np.concatenate((train_data1, test_data1), axis=0)
    data2 = np.concatenate((train_data2, test_data2), axis=0)
    data1 = min_max_normalize(data1)
    data2 = min_max_normalize(data2)
    label = np.concatenate((train_labels, test_labels), axis=0)
    data1 = torch.tensor(data1.reshape(-1, 1, 28, 28), dtype=torch.float32)
    data2 = torch.tensor(data2.reshape(-1, 1, 112, 112), dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)
    return data1, data2, label


def get_nyu_depth_v2(root):
    file_path = root + 'NYU-Depth-V2/NYU-Depth-V2_data.mat'
    Data = h5py.File(file_path, 'r')

    depths = Data['depths'][:]
    images = Data['images'][:]
    labels = Data['labels40'][:]
    # sceneTypes = Data['sceneTypes'][:]
    # scenes_ = [Data[scene] for scene in sceneTypes[0]]
    images = min_max_normalize(images)
    depths = min_max_normalize(depths)

    depths = torch.tensor(np.transpose(depths, (0, 2, 1)), dtype=torch.float32)  # [N, W, H]->[N, H, W]
    images = torch.tensor(np.transpose(images, (0, 1, 3, 2)), dtype=torch.float32)  # [N, C, W, H]->[N, C, H, W]
    labels = torch.tensor(np.transpose(labels, (0, 2, 1)).astype(np.int32), dtype=torch.long)  # [N, W, H]->[N, H, W]
    return images, depths.unsqueeze(1), labels


def get_ravdess(root):
    video_data = np.load(root + 'RAVDESS_preprocessed_npy/video_data.npy')
    audio_data = np.load(root + 'RAVDESS_preprocessed_npy/audio_data.npy')
    label_data = np.load(root + 'RAVDESS_preprocessed_npy/label_data.npy')
    video_data = min_max_normalize(video_data)
    audio_data = min_max_normalize(audio_data)
    video_data = torch.tensor(video_data, dtype=torch.float32)
    audio_data = torch.tensor(audio_data, dtype=torch.float32)
    label_data = torch.tensor(label_data, dtype=torch.long)
    return video_data, audio_data, label_data


def get_vggsound(rootpath):
    if platform.system() == 'Linux':
        root = rootpath + 'VGGS50K/'
    else:
        if os.environ['COMPUTERNAME'] == 'DESKTOP-A1JBA32':
            root = 'G:/Gray/Database_CMKD/VGGS50K/'
        else:
            root = 'D:/Gray/Database/VGGS50K/'
    vf_list, af_list, labels = [], [], []
    with open(root + 'VGGS50k_metadata.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = [line.strip() for line in lines]
        # data = data[:1000]  # For debug(failed for contrastive learning)
        for line in tqdm(data, total=len(data), desc='Data loading...'):
            sample_name = line.split('&')[0]
            label = int(line.split('&')[1])
            vf = root + 'VGGS50K_features_normed/visual_features/' + f'{sample_name}_vFeature.npy'
            af = root + 'VGGS50K_features_normed/audio_features/' + f'{sample_name}_aFeature.npy'
            avc_label = np.load(root + 'seg_labels/' + f'{sample_name}_sLabel.npy')
            slabel = _obtain_avel_label(avc_label, label)
            vf_list.append(torch.from_numpy(np.load(vf).astype(np.float32)))
            af_list.append(torch.from_numpy(np.load(af)))
            labels.append(torch.from_numpy(slabel.astype(np.float32)))
    return vf_list, af_list, labels


def _obtain_avel_label(avc_label, class_id):
    # avc_label: [1, 10]
    T, category_num = 10, 141
    label = np.zeros((T, category_num + 2))  # add 'background' category [10, 141+1] changed by Gray: [10, 141+1+1]
    bg_flag = 1 - avc_label
    label[:, class_id] = avc_label
    label[:, -2] = bg_flag
    label[:, -1] = class_id
    return label


def get_cmmd_v2(root):
    visual_features, text_features, labels = torch.load(root + 'CMMD-V2/CMMD_data.pth', weights_only=False)
    # visual_features = min_max_normalize(visual_features)
    # text_features = min_max_normalize(text_features)
    visual_features = torch.tensor(visual_features, dtype=torch.float32)
    text_features = torch.tensor(text_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return visual_features, text_features, labels


def metrics(prediction, target, n_classes, ignored_labels=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    prediction = np.array(prediction)
    target = np.array(target)

    ignored_labels = [] if ignored_labels is None else [ignored_labels]
    ignored_mask = ~np.isin(target, ignored_labels)
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    # Compute confusion matrix
    valid_labels = [label for label in range(n_classes) if label not in ignored_labels]
    cm = confusion_matrix(target, prediction, labels=valid_labels)
    results["Confusion matrix"] = cm

    # Compute Overall Accuracy (OA)
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy /= float(total)
    accuracy = Decimal(str(accuracy))
    results["Accuracy"] = accuracy.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

    # Compute F1 scores
    F1scores = f1_score(target, prediction, labels=valid_labels, average='weighted')
    F1scores = Decimal(str(F1scores))
    results["F1 scores"] = F1scores.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

    # Compute Precision for every class
    Precisions = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            Precision = 1. * cm[i, i] / np.sum(cm[:, i])
        except ZeroDivisionError:
            Precision = 0.
        Precision = Decimal(str(Precision))
        Precisions[i] = Precision.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    results["Precisions"] = Precisions

    # Compute Average Accuracy (AA)
    AAs = []
    for i in range(len(cm)):
        try:
            recall = cm[i][i] / np.sum(cm[i, :])
            if np.isnan(recall):
                continue
        except ZeroDivisionError:
            recall = 0.
        AAs.append(recall)
    avg_recall = Decimal(str(np.mean(AAs)))
    results['AA'] = avg_recall.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

    # Compute Kappa coefficient
    kappa = cohen_kappa_score(target, prediction, labels=valid_labels)
    kappa = Decimal(str(kappa))
    results["Kappa"] = kappa.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

    # Compute mean IoU
    num_classes = cm.shape[0]  # 已排除忽略项
    iou_scores = []
    nanmean = False
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        # 计算 IoU
        intersection = TP
        union = TP + FP + FN
        if union == 0:
            iou_score = np.nan  # 如果没有样本的情况下，IoU 为 NaN
            nanmean = True
        else:
            iou_score = intersection / union
        iou_scores.append(iou_score)
    mIoU = np.mean(iou_scores) if not nanmean else np.nanmean(iou_scores)
    mIoU = Decimal(str(mIoU))
    results["mIoU"] = mIoU.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    return results


def get_Tmodules(args, device):
    model_t = None
    optimizer_t = None
    scheduler_t = None
    criterion, metric = None, None
    preprocessing, postprocessing = None, None
    if args.database == 'AV-MNIST':
        if args.Tmodel == 'CNN-I':
            model_t = IntermediateFusionNet()
        elif args.Tmodel == 'ThreeLayerCNN-A':
            model_t = ThreeLayerCNN_A(cls_num=10)
        elif args.Tmodel == 'LeNet5':
            model_t = LeNet5(cls_num=10)
        else:
            raise ValueError(f'Invalid Tmodel: {args.Tmodel} for AV-MNIST')
        optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
        scheduler_t = optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

    elif args.database == 'NYU-Depth-V2':
        if args.Tmodel == 'FuseNet-I':
            mode = args.Tmodel.split('-')[1]
            model_t = define_FuseNet(num_labels=41, mode=mode)
            optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
            scheduler_t = optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=args.epochs)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        elif args.Tmodel == 'FuseNet-RGBbranch':
            model_t = define_branches(num_labels=41, mode='RGB')
            optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        elif args.Tmodel == 'FuseNet-Dbranch':
            model_t = define_branches(num_labels=41, mode='D')
            optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        else:
            raise ValueError(f'Invalid Tmodel: {args.Tmodel} for NYU-Depth-V2')

    elif args.database == 'RAVDESS':
        if args.Tmodel == 'DSCNN-I':
            model_t = DualStreamCNN(cls_num=8)
            optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
            scheduler_t = optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=args.epochs)
            criterion = nn.CrossEntropyLoss()
            preprocessing = DSCNN_preprocessing()
        elif args.Tmodel == 'AudioBranchNet':
            model_t = AudioBranchNet(cls_num=8)
            optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()
        elif args.Tmodel == 'VisualBranchNet':
            model_t = VisualBranchNet(cls_num=8)
            optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
            preprocessing = VisualBranchNet_preprocessing()
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Invalid Tmodel: {args.Tmodel} for RAVDESS')

    elif args.database == 'CMMD-V2':
        if args.Tmodel == 'MLP-I':
            model_t = IntermediateFusionMLP()
        elif args.Tmodel == 'MLP-Vb':
            model_t = visualMLP()
        elif args.Tmodel == 'MLP-Tb':
            model_t = textualMLP()
        else:
            raise ValueError(f'Invalid Tmodel: {args.Tmodel} for CMMD-V2')
        optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
        scheduler_t = optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

    elif args.database == 'VGGSound-50k':
        if args.Tmodel == 'DSCNN-VGGS-I':
            model_t = DualStreamCNN_VGGS(cls_num=141)
            optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
            scheduler_t = optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=args.epochs)
            criterion = nn.CrossEntropyLoss()
            preprocessing = DSCNN_VGGS_preprocessing()
            postprocessing = DSCNN_VGGS_postprocessing(criterion)
        elif args.Tmodel == 'AudioBranchNet-VGGS':
            model_t = AudioBranchNet_VGGS(cls_num=141)
            criterion = nn.CrossEntropyLoss()
            optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
            postprocessing = event_label_postprocessing(criterion)
        elif args.Tmodel == 'VisualBranchNet-VGGS':
            model_t = VisualBranchNet_VGGS(cls_num=141)
            preprocessing = VisualBranchNet_VGGS_preprocessing()
            criterion = nn.CrossEntropyLoss()
            optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr)
            postprocessing = event_label_postprocessing(criterion)
        else:
            raise ValueError(f'Invalid Tmodel: {args.Tmodel} for VGGSound-50k')
    metric = model_metrics(args, model_name=args.Tmodel)
    optimizers = [optimizer_t]
    return model_t, optimizers, scheduler_t, criterion, preprocessing, postprocessing, metric


def optimizers_zero_grad(optimizers):
    for optimizer in optimizers:
        if isinstance(optimizer, list):
            for opts in optimizer:
                for opt in opts:
                    opt.zero_grad()
        else:
            optimizer.zero_grad()


def optimizers_step(optimizers, epoch):
    for optimizer in optimizers:
        if isinstance(optimizer, list):
            if epoch < 100:
                optimizer = optimizer[0]
            elif 100 < epoch < 200:
                optimizer = optimizer[1]
            else:
                optimizer = optimizer[2]
            for opt in optimizer:
                opt.step()
        else:
            optimizer.step()


def L1_penalty(var):
    return torch.abs(var).sum()


class SAFN_preprocessing:
    def __init__(self):
        super(SAFN_preprocessing, self).__init__()

    def __call__(self, model, data1, data2):
        data2 = data2.reshape(data2.shape[0] * data2.shape[1], data2.shape[2], data2.shape[3], data2.shape[4])
        return model(data1, data2)


class DSCNN_preprocessing:
    def __init__(self):
        super(DSCNN_preprocessing, self).__init__()

    def __call__(self, model, data1, data2):
        data1 = data1.permute(0, 2, 1, 3, 4)
        return model(data1, data2)


class DSCNN_VGGS_preprocessing:
    def __init__(self):
        super(DSCNN_VGGS_preprocessing, self).__init__()

    def __call__(self, model, data1, data2):
        data1 = data1.permute(0, 4, 1, 2, 3)
        return model(data1, data2)


class DSCNN_VGGS_postprocessing:
    def __init__(self, criterion):
        super(DSCNN_VGGS_postprocessing, self).__init__()
        self.criterion = criterion

    def __call__(self, output, target):
        target = target[:, 0, -1].long()
        return self.criterion(output, target)


class VisualBranchNet_preprocessing:
    def __init__(self):
        super(VisualBranchNet_preprocessing, self).__init__()

    def __call__(self, model, data):
        data = data.permute(0, 2, 1, 3, 4)
        return model(data)


class VisualBranchNet_VGGS_preprocessing:
    def __init__(self):
        super(VisualBranchNet_VGGS_preprocessing, self).__init__()

    def __call__(self, model, data):
        data = data.permute(0, 4, 1, 2, 3)
        return model(data)


class CPSP_postprocessing(nn.Module):
    def __init__(self, criterion_seg, criterion_event, device, mode=None):
        super(CPSP_postprocessing, self).__init__()
        self.loss1 = criterion_seg
        self.loss2 = criterion_event
        self.device = device
        self.mode = mode  # avps, scon, vcon

    def forward(self, output, target):
        target = target[:, :, :-1]
        is_event_scores, event_scores, avps, fusion = output
        """some processing on the labels"""
        labels_foreground = target[:, :, :-1]  # [B, T, cls_num]
        labels_BCE, labels_evn = labels_foreground.max(-1)  # [B, 10], [B, 10]
        labels_event, _ = labels_evn.max(-1)  # [B]

        event_flag, pos_flag, neg_flag = get_flag_by_gt(labels_BCE)
        event_class_flag = labels_event

        """compute loss and backward"""
        loss_is_event = self.loss1(is_event_scores, labels_BCE.to(self.device))
        loss_event_class = self.loss2(event_scores, labels_event.to(self.device))
        loss = loss_is_event + loss_event_class

        if self.mode == 'avps':
            lambda_avps = 100
            soft_labels = labels_BCE / (labels_BCE.sum(-1, keepdim=True) + 1e-6)
            loss_avps = AVPSLoss(avps, soft_labels)
            loss += lambda_avps * loss_avps
        if self.mode == 'scon':
            lambda_scon = 0.01
            loss_scon = segment_contrastive_loss(fusion, event_flag, pos_flag, neg_flag)
            loss += lambda_scon * loss_scon
        if self.mode == 'vcon':
            lambda_vcon = 1
            loss_vcon = video_contrastive_loss(fusion, event_class_flag)
            loss += lambda_vcon * loss_vcon
        return loss


class event_label_postprocessing(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, outputs, target):
        target = target[:, 0, -1].long()
        loss = self.criterion(outputs, target)
        return loss


class GrayIm_preprocessing:
    def __init__(self):
        super(GrayIm_preprocessing, self).__init__()

    def __call__(self, model, data):
        data = data.repeat(1, 3, 1, 1) if data.shape[1] == 1 else data
        return model(data)


class GrayIm_resize_preprocessing:
    def __init__(self, tar_size=224):
        super(GrayIm_resize_preprocessing, self).__init__()
        self.tar_size = tar_size

    def __call__(self, model, data):
        data = data.repeat(1, 3, 1, 1)
        data = F.interpolate(data, size=(self.tar_size, self.tar_size), mode='bilinear', align_corners=False)
        return model(data)


def get_flag_by_gt(is_event_scores):
    # is_event_scores: [bs, 10]
    scores_pos_ind = is_event_scores  # > 0.5
    pred_temp = scores_pos_ind.long()  # [B, 10]
    pred = pred_temp.unsqueeze(1)  # [B, 1, 10]
    pos_flag = pred.repeat(1, 10, 1)  # [B, 10, 10]
    pos_flag *= pred.permute(0, 2, 1)
    neg_flag = (1 - pred).repeat(1, 10, 1)  # [B, 10, 10]
    neg_flag *= pred.permute(0, 2, 1)
    return pred_temp, pos_flag, neg_flag


def AVPSLoss(av_simm, soft_label):
    """audiovisual pair similarity loss for fully supervised setting,
    please refer to Eq.(8, 9) in our paper.
    """
    # av_simm: [bs, 10]
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss


def segment_contrastive_loss(fusion, frame_corr_one_hot, segment_flag_pos, segment_flag_neg, t=0.6):
    num_event = torch.sum(frame_corr_one_hot, dim=-1, keepdim=True)  # [bs, 1]
    all_bg_flag = (num_event != 0).to(torch.float32)
    num_bg = 10 - num_event
    fusion = F.normalize(fusion, dim=-1)  # [bs, 10, 256]
    cos_simm = torch.bmm(fusion, fusion.permute(0, 2, 1))  # [bs, 10, 10]

    mask_simm_neg = cos_simm * segment_flag_neg
    mask_simm_neg /= t
    mask_simm_exp_neg = torch.exp(mask_simm_neg) * segment_flag_neg

    mask_simm_pos = cos_simm * segment_flag_pos
    mask_simm_pos /= t
    mask_simm_exp_pos = torch.exp(mask_simm_pos) * segment_flag_pos

    simm_pos = torch.sum(mask_simm_exp_pos, dim=1)  # [bs, 10], column-summation
    avg_simm_all_negs = torch.sum(mask_simm_exp_neg, dim=1) / (num_bg + 1e-12)  # [bs, 10]
    simm_pos_all_negs = simm_pos + avg_simm_all_negs
    temp_result = simm_pos / (simm_pos_all_negs + 1e-12)
    loss = torch.sum((-1) * torch.log(temp_result + 1e-12), dim=-1, keepdim=True) / (num_event + 1e-12)  # [bs, 1]
    # pdb.set_trace()
    loss *= all_bg_flag
    loss = torch.sum(loss) / torch.sum(all_bg_flag)
    return loss


def video_contrastive_loss(fusion, batch_class_labels, margin=0.2, neg_num=3):
    """loss_vpsa used in PSA_V of CPSP"""
    # fusion: [bs, 10, dim=256], batch_class_labels: [bs,]
    fusion = F.normalize(fusion, dim=-1)
    avg_fea = torch.mean(fusion, dim=1)  # [bs, 256]
    bs = avg_fea.size(0)
    dist = torch.pow(avg_fea, 2).sum(dim=1, keepdim=True).expand(bs, bs)
    dist = dist + dist.t()
    dist.addmm_(1, -2, avg_fea, avg_fea.t())
    dist = dist.clamp(min=1e-12).sqrt()

    mask = batch_class_labels.expand(bs, bs).eq(batch_class_labels.expand(bs, bs).t()).float()

    INF = 1e12
    NEG_INF = (-1) * INF

    dist_ap = dist * mask
    dist_an = dist * (1 - mask)

    dist_ap += torch.ones_like(dist_ap) * NEG_INF * (1 - mask)
    dist_an += torch.ones_like(dist_an) * INF * mask

    topk_dist_ap, topk_ap_indices = dist_ap.topk(k=1, dim=1, largest=True, sorted=True)
    topk_dist_an, topk_an_indices = dist_an.topk(k=neg_num, dim=1, largest=False, sorted=True)

    avg_topk_dist_ap = topk_dist_ap.squeeze(-1)
    avg_topk_dist_an = topk_dist_an.mean(dim=-1)
    y = torch.ones_like(avg_topk_dist_an)

    loss = nn.MarginRankingLoss(margin=margin)(avg_topk_dist_an, avg_topk_dist_ap, y)
    return loss


def LabelFreeSelfSupervisedNCELoss(fea_a, fea_v, t=0.1):
    bs, seg_num, dim = fea_a.shape
    fea_a = F.normalize(fea_a, dim=-1)  # [bs, 10, 256]
    fea_v = F.normalize(fea_v, dim=-1)  # [bs, 10, 256]
    rs_fea_a = fea_a.reshape(-1, dim)  # [bs*10, 256]
    rs_fea_v = fea_v.reshape(-1, dim)  # [bs*10, 256]
    pos_av_simm = torch.sum(torch.mul(rs_fea_a, rs_fea_v), dim=-1)  # [bs*10=N,]
    batch_simm = torch.mm(rs_fea_a, rs_fea_v.t())  # [N, N]

    pos_av_simm /= t
    batch_simm /= t
    pos_av_simm = torch.exp(pos_av_simm)
    batch_simm = torch.exp(batch_simm)

    pos1_neg_item = torch.sum(batch_simm, dim=-1)  # [N,]
    loss1 = torch.mean((-1) * torch.log(pos_av_simm / pos1_neg_item))

    return loss1


def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        n_total_params += n_elem
    return n_total_params


def min_max_normalize(x):
    if x.ndim == 4:
        # x shape: [N, C, H, W]
        min_vals = x.min(axis=(0, 2, 3), keepdims=True)  # [1, C, 1, 1]
        max_vals = x.max(axis=(0, 2, 3), keepdims=True)  # [1, C, 1, 1]
    elif x.ndim == 3:
        # x shape: [N, H, W]
        min_vals = x.min(axis=(1, 2), keepdims=True)  # [N, 1, 1]
        max_vals = x.max(axis=(1, 2), keepdims=True)  # [N, 1, 1]
    elif x.ndim == 5:
        # x shape: [N, T, C, H, W]
        min_vals = x.min(axis=(0, 1, 3, 4), keepdims=True)  # [1, 1, C, 1, 1]
        max_vals = x.max(axis=(0, 1, 3, 4), keepdims=True)  # [1, 1, C, 1, 1]
    elif x.ndim == 2:
        # x shape: [N, M]
        min_vals = x.min(axis=1, keepdims=True)  # [N, 1]
        max_vals = x.max(axis=1, keepdims=True)  # [N, 1]
    elif x.ndim == 1:
        # x shape: [M]
        min_vals = x.min(keepdims=True)  # [1]
        max_vals = x.max(keepdims=True)  # [1]
    else:
        raise ValueError("Dimension error!")

    x_normalized = (x - min_vals) / (max_vals - min_vals)

    return x_normalized


def normalize_tensor_global(tensor_list):
    all_values = np.concatenate([tensor.flatten() for tensor in tensor_list])
    global_min = all_values.min()
    global_max = all_values.max()

    normalized_tensors = []
    for tensor in tqdm(tensor_list, total=len(tensor_list), desc='Data Normalization...'):
        normalized_tensor = (tensor - global_min) / (global_max - global_min)
        normalized_tensors.append(normalized_tensor)

    return normalized_tensors


class model_metrics:
    def __init__(self, args, model_name=None):
        self.model = model_name
        if args.database == 'NYU-Depth-V2':
            self.n_classes = 41
            self.ignored_labels = 0
        elif args.database == 'RAVDESS':
            self.n_classes = 8
            self.ignored_labels = None
        elif args.database == 'AV-MNIST':
            self.n_classes = 10
            self.ignored_labels = None
        elif args.database == 'VGGSound-50k':
            self.n_classes = 141
            self.ignored_labels = None
        elif args.database == 'CMMD-V2':
            self.n_classes = 8
            self.ignored_labels = None
        else:
            raise ValueError("Unsupported database for metrics calculation.")
        self.reset()

    def reset(self):
        self.pred_list = []  # [B, Cls, H, W] ([[B, Cls, H, W], [B, Cls, H, W]])
        self.gt_list = []  # [B, Cls, H, W]

    def update(self, pred, gt):
        if self.model in ['FuseNet-RGBbranch', 'FuseNet-Dbranch']:
            output = pred
            p = torch.max(output, dim=1)[1]
            self.pred_list.extend(p.cpu().flatten().detach().tolist())
            self.gt_list.extend(gt.cpu().flatten().detach().tolist())

        elif self.model in ['DSCNN-VGGS-I', 'DSCNN-VGGS-L']:
            gt = gt[:, 0, -1].long()
            p = torch.max(pred, dim=1)[1]
            self.pred_list.extend(p.cpu().flatten().detach().tolist())
            self.gt_list.extend(gt.cpu().flatten().detach().tolist())

        elif self.model in ['VisualBranchNet-VGGS', 'AudioBranchNet-VGGS']:
            gt = gt[:, 0, -1].long()
            p = torch.max(pred, dim=1)[1]
            self.pred_list.extend(p.cpu().flatten().detach().tolist())
            self.gt_list.extend(gt.cpu().flatten().detach().tolist())

        else:
            p = torch.max(pred, dim=1)[1]
            self.pred_list.extend(p.cpu().flatten().detach().tolist())
            self.gt_list.extend(gt.cpu().flatten().detach().tolist())

    def compute(self):
        return metrics(self.pred_list, self.gt_list, n_classes=self.n_classes, ignored_labels=self.ignored_labels)


def get_Smodules(args):
    model_s = None
    ignored_labels = None
    n_classes, criterion = None, None
    preprocessing, postprocessing = None, None

    if args.database == 'AV-MNIST':
        n_classes = 10
        if args.Smodel == 'ThreeLayerCNN-A':
            model_s = ThreeLayerCNN_A(cls_num=n_classes)
        elif args.Smodel == 'LeNet5':
            model_s = LeNet5(cls_num=n_classes)
        else:
            preprocessing = GrayIm_preprocessing()
        criterion = nn.CrossEntropyLoss()

    elif args.database == 'CMMD-V2':
        n_classes = 8
        if args.Smodel == 'MLP-Vb':
            model_s = visualMLP()
        elif args.Smodel == 'MLP-Tb':
            model_s = textualMLP()
        criterion = nn.CrossEntropyLoss()

    elif args.database == 'NYU-Depth-V2':
        n_classes = 41
        ignored_labels = 0
        if args.Smodel == 'FuseNet-RGBbranch':
            model_s = define_branches(num_labels=n_classes, mode='RGB')
            criterion = nn.CrossEntropyLoss(ignore_index=ignored_labels)
        elif args.Smodel == 'FuseNet-Dbranch':
            model_s = define_branches(num_labels=n_classes, mode='D')
            criterion = nn.CrossEntropyLoss(ignore_index=ignored_labels)

    elif args.database == 'RAVDESS':
        n_classes = 8
        if args.Smodel == 'AudioBranchNet':
            model_s = AudioBranchNet(cls_num=n_classes)
        elif args.Smodel == 'VisualBranchNet':
            model_s = VisualBranchNet(cls_num=n_classes)
            preprocessing = VisualBranchNet_preprocessing()
        criterion = nn.CrossEntropyLoss()

    elif args.database == 'VGGSound-50k':
        n_classes = 141
        if args.Smodel == 'AudioBranchNet-VGGS':
            model_s = AudioBranchNet_VGGS(cls_num=n_classes)
        elif args.Smodel == 'VisualBranchNet-VGGS':
            model_s = VisualBranchNet_VGGS(cls_num=n_classes)
            preprocessing = VisualBranchNet_VGGS_preprocessing()
        criterion = nn.CrossEntropyLoss()
        postprocessing = event_label_postprocessing(criterion)

    optimizer_s = optim.Adam(model_s.parameters(), lr=args.lr)
    scheduler_s = optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=args.epochs)
    metric = model_metrics(args, model_name=args.Smodel)
    return model_s, optimizer_s, scheduler_s, criterion, preprocessing, postprocessing, metric


def get_AUXmodules(args):
    model_s = None
    ignored_labels = None
    n_classes, criterion = None, None
    preprocessing, postprocessing = None, None

    if args.database == 'AV-MNIST':
        n_classes = 10
        if args.AUXmodel == 'ThreeLayerCNN-A':
            model_s = ThreeLayerCNN_A(cls_num=n_classes)
        elif args.AUXmodel == 'LeNet5':
            model_s = LeNet5(cls_num=n_classes)
        else:
            preprocessing = GrayIm_preprocessing()
        criterion = nn.CrossEntropyLoss()

    elif args.database == 'NYU-Depth-V2':
        n_classes = 41
        ignored_labels = 0
        if args.AUXmodel == 'FuseNet-RGBbranch':
            model_s = define_branches(num_labels=n_classes, mode='RGB')
            criterion = nn.CrossEntropyLoss(ignore_index=ignored_labels)
        elif args.AUXmodel == 'FuseNet-Dbranch':
            model_s = define_branches(num_labels=n_classes, mode='D')
            criterion = nn.CrossEntropyLoss(ignore_index=ignored_labels)

    elif args.database == 'RAVDESS':
        n_classes = 8
        if args.AUXmodel == 'AudioBranchNet':
            model_s = AudioBranchNet(cls_num=n_classes)
        elif args.AUXmodel == 'VisualBranchNet':
            model_s = VisualBranchNet(cls_num=n_classes)
            preprocessing = VisualBranchNet_preprocessing()
        criterion = nn.CrossEntropyLoss()

    elif args.database == 'VGGSound-50k':
        n_classes = 141
        if args.AUXmodel == 'AudioBranchNet-VGGS':
            model_s = AudioBranchNet_VGGS(cls_num=n_classes)
        elif args.AUXmodel == 'VisualBranchNet-VGGS':
            model_s = VisualBranchNet_VGGS(cls_num=n_classes)
            preprocessing = VisualBranchNet_VGGS_preprocessing()
        criterion = nn.CrossEntropyLoss()
        postprocessing = event_label_postprocessing(criterion)

    elif args.database == 'CMMD-V2':
        n_classes = 8
        if args.AUXmodel == 'MLP-Vb':
            model_s = visualMLP()
        elif args.AUXmodel == 'MLP-Tb':
            model_s = textualMLP()
        criterion = nn.CrossEntropyLoss()

    optimizer_s = optim.Adam(model_s.parameters(), lr=args.lr)
    scheduler_s = optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=args.epochs)
    metric = model_metrics(args, model_name=args.AUXmodel)
    return model_s, optimizer_s, scheduler_s, criterion, preprocessing, postprocessing, metric


def hook_fn(name, features):
    def hook(module, input, output):
        features[name] = output
    return hook


def get_submodule(model, submodule_name):
    names = submodule_name.split('.')
    submodule = model
    for name in names:
        submodule = submodule._modules[name]
    return submodule


def hooks_builder(model, hook_names):
    features = {}
    hooks = []
    for name in hook_names:
        submodule = get_submodule(model, name)
        hook = submodule.register_forward_hook(hook_fn(name, features))
        hooks.append(hook)
    return hooks, features


def hook_builder(model, hook_name):
    features = {}
    hooks = []
    submodule = get_submodule(model, hook_name)
    hook = submodule.register_forward_hook(hook_fn(hook_name, features))
    hooks.append(hook)
    return hooks, features


def hook_fn_grad(name, gradients):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output
    return hook


def hooks_builder_grad(model, hook_names):
    gradients = {}
    hooks = []
    for name in hook_names:
        submodule = get_submodule(model, name)
        hook = submodule.register_full_backward_hook(hook_fn_grad(name, gradients))
        hooks.append(hook)
    return hooks, gradients


def hooks_remover(hooks):
    for hook in hooks:
        hook.remove()


def get_avg_grads(grads_t, MGrad_names):
    grad_m1 = grads_t[MGrad_names[0]][0]
    avg_grad_m1 = torch.mean(grad_m1, dim=0)
    grad_m2 = grads_t[MGrad_names[1]][0]
    avg_grad_m2 = torch.mean(grad_m2, dim=0)
    return avg_grad_m1, avg_grad_m2


def get_dataset(database_name, data, phase, exp_id):
    indices = get_indices(database_name, phase, exp_id)
    data = [data[0][i] for i in indices], [data[1][i] for i in indices], [data[2][i] for i in indices]
    return data


def get_indices(database_name, phase, exp_id):
    random_indices = pd.read_csv(f"./metadata/{database_name}_indices.csv")
    indices = random_indices[f'group_{exp_id}']
    num_data = len(indices)
    if phase == 'train':
        return indices[:int(num_data * 0.6)]
    elif phase == 'val':
        return indices[int(num_data * 0.6):int(num_data * 0.8)]
    elif phase == 'test':
        return indices[int(num_data * 0.8):]
    else:
        raise ValueError(f"Invalid phase: {phase}")


def data_preprocessing(data1, data2, data_mode):
    if data_mode in ['ORG', 'm1', 'm2']:
        return data1, data2
    else:
        raise ValueError(f"Invalid data mode: {data_mode}. Supported modes are 'ORG', 'm1', 'm2'.")


if __name__ == '__main__':
    res = get_data('CMMD-V2')
    print(res)