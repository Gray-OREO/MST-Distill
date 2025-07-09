import torch
from torch.utils.data import DataLoader
from scipy.stats import kendalltau
import os
from utils import seed_all, get_data, get_dataset, get_Tmodules, get_Smodules, hooks_builder, hooks_remover
from KD_methods.C2KD import get_C2KDmodules, ntkl, ntkl_ss, get_feats, get_logits, get_labels
import copy
import numpy as np
import torch.nn.functional as F
import time
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Dataset import MultiModalX
import sys


def batch_kendall_tau_gpu(logits_t, logits_s):
    """
    input：
        logits_t: (batch_size, dim)
        logits_s: (batch_size, dim)
    output：
        tau_values: (batch_size,)
    """
    device = logits_t.device
    batch_size, n = logits_t.shape

    i, j = torch.triu_indices(n, n, offset=1)
    i, j = i.to(device), j.to(device)

    x1 = logits_t[:, i]  # (batch, pair)
    x2 = logits_t[:, j]
    y1 = logits_s[:, i]
    y2 = logits_s[:, j]

    sign_x = torch.sign(x1 - x2)
    sign_y = torch.sign(y1 - y2)

    product = sign_x * sign_y
    C = (product > 0).sum(dim=1).float()
    D = (product < 0).sum(dim=1).float()

    tie_x = (sign_x == 0).sum(dim=1).float()
    tie_y = (sign_y == 0).sum(dim=1).float()

    total_pairs = n * (n-1) / 2
    adjust_x = total_pairs - tie_x
    adjust_y = total_pairs - tie_y

    denominator = torch.sqrt(adjust_x * adjust_y)
    tau = (C - D) / (denominator + 1e-8)  # 防止除零

    return tau


def image_level_kendalltau(logits_t, logits_s, original_shape, chunk_size=131072):
    """
    图像级别Kendall Tau-b计算（像素级平均）
    输入：
        logits_t: (N, 41) 经过view处理的张量 (N = batch_size*h*w)
        logits_s: (N, 41)
        original_shape: 原始空间形状 (batch_size, h, w)
        chunk_size: 分块大小（推荐显存的1/4）
    输出：
        batch_tau: (batch_size,)
    """
    device = logits_t.device
    batch_size, h, w = original_shape
    n_pixels = batch_size * h * w
    cls_num = logits_t.size(1)

    i, j = torch.triu_indices(cls_num, cls_num, offset=1)
    i, j = i.to(device), j.to(device)
    pair_count = i.size(0)

    tau_accum = torch.zeros(batch_size, device=device)
    count_accum = torch.zeros(batch_size, device=device)

    for chunk_start in range(0, n_pixels, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_pixels)

        chunk_t = logits_t[chunk_start:chunk_end]  # (chunk_size, 41)
        chunk_s = logits_s[chunk_start:chunk_end]

        batch_indices = torch.arange(chunk_start, chunk_end, device=device) // (h * w)

        x1 = chunk_t[:, i]  # (chunk, 820)
        x2 = chunk_t[:, j]
        y1 = chunk_s[:, i]
        y2 = chunk_s[:, j]

        sign_x = (x1 > x2).float() - (x1 < x2).float()  # 比torch.sign节省30%显存
        sign_y = (y1 > y2).float() - (y1 < y2).float()

        product = sign_x * sign_y
        C = (product > 0).sum(dim=1).float()
        D = (product < 0).sum(dim=1).float()

        tie_x = (sign_x == 0).sum(dim=1).float()
        tie_y = (sign_y == 0).sum(dim=1).float()

        total = cls_num * (cls_num - 1) / 2
        denominator = torch.sqrt((total - tie_x) * (total - tie_y))
        tau = (C - D) / (denominator + 1e-8)

        for b in range(batch_size):
            mask = (batch_indices == b)
            if mask.any():
                tau_accum[b] += tau[mask].sum()
                count_accum[b] += mask.sum()

    batch_tau = tau_accum / (count_accum + 1e-8)
    return batch_tau


def expand_mask(mask, spatial_shape):
    """
    将batch级别的mask扩展到像素级别
    输入：
        mask: (batch_size,)
        spatial_shape: (h, w)
    输出：
        expanded_mask: (batch_size*h*w, 1)
    """
    h, w = spatial_shape
    # 扩展维度：batch_size -> batch_size, h*w
    expanded = mask.unsqueeze(1).expand(-1, h*w)  # (batch_size, h*w)
    # 转置并展平
    return expanded.contiguous().view(-1, 1)  # (batch_size*h*w, 1)


if __name__ == '__main__':
    '''
    Args Setting for CML.
    '''
    parser = ArgumentParser(description='C2KD')
    parser.add_argument('--database', type=str, default='AV-MNIST',
                        help="database name must be one of "
                             "['NYU-Depth-V2', 'RAVDESS', 'AV-MNIST', 'VGGSound-50k', 'MM-IMDb']")
    parser.add_argument('--Tmodel', type=str, default='CNN-I',
                        help='Teacher model name')

    parser.add_argument('--Smodel', type=str, default='LeNet5',
                        help='Student model name')
    parser.add_argument('--mode', type=str, default='m1',
                        help='Data mode: m1 or m2')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for training')

    parser.add_argument('--ckpt_name', type=str,
                        default='',
                        help='The name of the weight to be loaded in ./checkpoints/stu')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers for DataLoader')
    parser.add_argument('--krc', type=float, default=0.,
                        help='Kendall Rank Correlation threshold')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for training')
    parser.add_argument('--record', type=bool, default=False,
                        help='flag whether to record the learning log')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='cuda id')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs for training')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='flag whether to save best model')
    parser.add_argument('--test_phase', type=bool, default=False,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--commit', type=str, default='C2KD-baseline',
                        help='Commit for logs')
    args = parser.parse_args()

    seed_all(args.seed)

    # 保存log
    log_dir = f'./logs/c2kd'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = log_dir + f'/{args.database}_{args.lr}_{str(time.time()).split(".")[0]}_{args.commit}'
    writer = SummaryWriter(log_dir, write_to_disk=args.record)

    data = get_data(args.database)
    data_train = get_dataset(args.database, data, 'train', args.seed)
    data_val = get_dataset(args.database, data, 'val', args.seed)
    data_test = get_dataset(args.database, data, 'test', args.seed)

    train_dataset = MultiModalX(data_train, args.database, mode=args.mode)
    valid_dataset = MultiModalX(data_val, args.database, mode=args.mode)
    test_dataset = MultiModalX(data_test, args.database, mode=args.mode)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        pin_memory=True,
        batch_size=args.batch_size
    )

    test_loader = DataLoader(
        test_dataset,
        pin_memory=True,
        batch_size=args.batch_size
    )

    # ===========GPU Setting====================
    device = torch.device(f"cuda:{args.cuda_id}")
    # ==========Initialization===========
    model_t, _, scheduler_t, criterion_t, preprocessing_t, postprocessing_t, _ = get_Tmodules(args, device)
    model_s, _, scheduler_s, criterion_s, preprocessing_s, postprocessing_s, metric = get_Smodules(args)
    feat_names, px_t, px_s, KL_batchmean, KL_none = get_C2KDmodules(args)
    optimizer = torch.optim.SGD(list(model_t.parameters()) + list(model_s.parameters())
                                + list(px_t.parameters()) + list(px_s.parameters()), lr=args.lr, momentum=0.9)

    model_t = model_t.to(device)
    model_s = model_s.to(device)
    px_t = px_t.to(device)
    px_s = px_s.to(device)
    model_t.load_state_dict(torch.load(f'checkpoints/stu/wo_kd/{args.ckpt_name}', map_location=device, weights_only=False))

    # C2KD After Model Pretraining
    best_model_state = None
    best_val_loss = float('inf')

    print('================= C2KD =================')
    for epoch in range(args.epochs):
        start_time1 = time.time()
        # train
        model_t.train()
        model_s.train()
        px_t.train()
        px_s.train()

        tra_LOSS_s = 0
        for i, (data, data2, label) in tqdm(enumerate(train_loader), desc="Model Training ...",
                                            total=len(train_loader), dynamic_ncols=True,
                                            disable=False, file=sys.stdout):
            # print('Train Iter {}'.format(i))
            data, data2, label = data.to(device), data2.to(device), label.to(device)
            if args.mode == 'm1':
                data_t, data_s = data2, data
            else:
                data_t, data_s = data, data2

            hooks_t, features_t = hooks_builder(model_t, feat_names[0])
            hooks_s, features_s = hooks_builder(model_s, feat_names[1])

            outputs_t = model_t(data_t) if preprocessing_t is None else preprocessing_t(model_t, data_t)
            outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

            logits_t = px_t(get_feats(args, features_t, *feat_names[0]))
            logits_s = px_s(get_feats(args, features_s, *feat_names[1]))

            pseu_label, outputs_, labels = get_logits(args, outputs_t), get_logits(args, outputs_s), get_labels(args, label)

            loss_t = criterion_t(outputs_t, label) if postprocessing_t is None else postprocessing_t(outputs_t, label)
            loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)

            if args.database == 'NYU-Depth-V2':
                logits_t=logits_t.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                logits_s=logits_s.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                pseu_label=pseu_label.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                outputs_=outputs_.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                labels=labels.permute(0, 2, 3, 1).contiguous().view(-1, 41)

            kl_loss_t_t_im = KL_batchmean(F.log_softmax(logits_t, -1), F.softmax(pseu_label.detach(), dim=-1))
            kl_loss_t_im_t = KL_batchmean(F.log_softmax(pseu_label, -1), F.softmax(logits_t.detach(), dim=-1))

            kl_loss_s_s_im = KL_batchmean(F.log_softmax(logits_s, -1), F.softmax(outputs_.detach(), dim=-1))
            kl_loss_s_im_s = KL_batchmean(F.log_softmax(outputs_, -1), F.softmax(logits_s.detach(), dim=-1))

            kendall = batch_kendall_tau_gpu(logits_t.detach(), logits_s.detach()) if args.database != 'NYU-Depth-V2' \
                else image_level_kendalltau(logits_t.detach(), logits_s.detach(), original_shape=(data.shape[0], 480, 640))
            kendall_mean = kendall.mean()
            mask = torch.where(kendall > args.krc, 1, 0)
            mask_num = mask.sum()
            del kendall

            if mask.sum()==0:
                kl_loss_st_s_t = 0
                kl_loss_st_t_s = 0
            else:
                if args.database == 'NYU-Depth-V2':
                    kl_loss_st_s_t = ntkl_ss(logits_s, logits_t.detach(), labels, (data.shape[0], 480, 640), mask, KL_batchmean)
                    kl_loss_st_t_s = ntkl_ss(logits_t, logits_s.detach(), labels, (data.shape[0], 480, 640), mask, KL_batchmean)
                else:
                    kl_loss_st_s_t = ntkl(logits_s, logits_t.detach(), labels, mask, KL_none)
                    kl_loss_st_t_s = ntkl(logits_t, logits_s.detach(), labels, mask, KL_none)

            hooks_remover(hooks_t)
            hooks_remover(hooks_s)
            loss = (loss_t + loss_s +
                    kl_loss_t_t_im + kl_loss_t_im_t +
                    kl_loss_s_s_im + kl_loss_s_im_s +
                    kl_loss_st_s_t + kl_loss_st_t_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tra_LOSS_s += loss.item()

        tra_LOSS_s_avg = tra_LOSS_s / (i + 1)
        print(f'Train =====> Epoch {epoch + 1}/{args.epochs}: loss = {tra_LOSS_s_avg:.4f}')
        writer.add_scalar('train/Loss', tra_LOSS_s_avg, epoch)

        # validation
        model_t.eval()
        model_s.eval()
        px_t.eval()
        px_s.eval()
        metric.reset()
        L_s_val = 0
        acc_c = 0
        gt_list, pred_list = [], []
        with torch.no_grad():
            for i, (data, data2, label) in tqdm(enumerate(valid_loader), desc="Model Validating ...",
                                                total=len(valid_loader), dynamic_ncols=True, disable=False,
                                                file=sys.stdout):
                # print('Val Iter {}'.format(i))
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                if args.mode == 'm1':
                    data_t, data_s = data2, data
                else:
                    data_t, data_s = data, data2

                # hooks_t, features_t = hooks_builder(model_t, feat_names[0])
                # hooks_s, features_s = hooks_builder(model_s, feat_names[1])

                # outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                # logits_t = px_t(get_feats(args, features_t, *feat_names[0]))
                # logits_s = px_s(get_feats(args, features_s, *feat_names[1]))

                # pseu_label, outputs_, labels = get_logits(outputs_t), get_logits(outputs_s), get_labels(args, label)
                #
                # loss_t = criterion_t(outputs_t, label) if postprocessing_t is None else postprocessing_t(outputs_t, label)
                loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)

                # if args.database == 'NYU-Depth-V2':
                #     logits_t = logits_t.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                #     logits_s = logits_s.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                #     pseu_label = pseu_label.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                #     outputs_ = outputs_.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                #     labels = labels.permute(0, 2, 3, 1).contiguous().view(-1, 41)

                # kl_loss_t_t_im = KL_batchmean(F.log_softmax(logits_t, -1), F.softmax(pseu_label.detach(), dim=-1))
                # kl_loss_t_im_t = KL_batchmean(F.log_softmax(pseu_label, -1), F.softmax(logits_t.detach(), dim=-1))
                #
                # kl_loss_s_s_im = KL_batchmean(F.log_softmax(logits_s, -1), F.softmax(outputs_.detach(), dim=-1))
                # kl_loss_s_im_s = KL_batchmean(F.log_softmax(outputs_, -1), F.softmax(logits_s.detach(), dim=-1))

                # kendall = np.empty(logits_t.size(0))
                # for i in range(logits_t.size(0)):
                #     temp = \
                #         kendalltau(logits_t[i, :].cpu().detach().numpy(), logits_s[i, :].cpu().detach().numpy())[0]
                #     kendall[i] = temp
                # kendall = torch.from_numpy(kendall).to(device)
                # kendall_mean = kendall.mean()
                # mask = torch.where(kendall > args.krc, 1, 0)
                # mask_num = kendall.sum()
                # del kendall

                # if mask.sum() == 0:
                #     kl_loss_st_s_t = 0
                #     kl_loss_st_t_s = 0
                # else:
                #     kl_loss_st_s_t = ntkl(logits_s, logits_t.detach(), labels, mask, KL_none)
                #     kl_loss_st_t_s = ntkl(logits_t, logits_s.detach(), labels, mask, KL_none)
                #
                # hooks_remover(hooks_t)
                # hooks_remover(hooks_s)

                # loss = (loss_t + loss_s +
                #     kl_loss_t_t_im + kl_loss_t_im_t +
                #     kl_loss_s_s_im + kl_loss_s_im_s +
                #     kl_loss_st_s_t + kl_loss_st_t_s)
                loss = loss_s
                L_s_val += loss.item()
                metric.update(outputs_s, label)
            L_s_val = L_s_val / (i + 1)
        # res = metric.compute()
        # print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_s_val:.4f}, OA = {float(res['Accuracy'])}")
        # writer.add_scalar('valid/Loss', L_s_val, epoch)
        # writer.add_scalar('valid/Acc', float(res['Accuracy']), epoch)

        # For NYU-Depth-V2
        print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_s_val:.4f}")
        writer.add_scalar('valid/Loss', L_s_val, epoch)

        # 保存验证集上表现最好的模型
        if L_s_val < best_val_loss:
            best_val_loss = L_s_val
            best_model_state = copy.deepcopy(model_s.state_dict())
            best_epoch = epoch

        # test
        if args.test_phase:
            model_t.eval()
            model_s.eval()
            px_t.eval()
            px_s.eval()
            metric.reset()
            L_t = 0
            acc_c = 0
            gt_list, pred_list = [], []
            with torch.no_grad():
                for i, (data, data2, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                                    total=len(test_loader), dynamic_ncols=True, disable=False,
                                                    file=sys.stdout):
                    data, data2, label = data.to(device), data2.to(device), label.to(device)
                    if args.mode == 'm1':
                        data_t, data_s = data2, data
                    else:
                        data_t, data_s = data, data2

                    hooks_t, features_t = hooks_builder(model_t, feat_names[0])
                    hooks_s, features_s = hooks_builder(model_s, feat_names[1])

                    outputs_t = model_t(data_t) if preprocessing_t is None else preprocessing_t(model_t, data_t)
                    outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                    logits_t = px_t(get_feats(args, features_t, *feat_names[0]))
                    logits_s = px_s(get_feats(args, features_s, *feat_names[1]))

                    pseu_label, outputs_, labels = get_logits(outputs_t), get_logits(outputs_s), get_labels(args, label)

                    loss_t = criterion_t(outputs_t, label) if postprocessing_t is None else postprocessing_t(outputs_t, label)
                    loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)

                    if args.database == 'NYU-Depth-V2':
                        logits_t = logits_t.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                        logits_s = logits_s.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                        pseu_label = pseu_label.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                        outputs_ = outputs_.permute(0, 2, 3, 1).contiguous().view(-1, 41)
                        labels = labels.permute(0, 2, 3, 1).contiguous().view(-1, 41)

                    kl_loss_t_t_im = KL_batchmean(F.log_softmax(logits_t, -1), F.softmax(pseu_label.detach(), dim=-1))
                    kl_loss_t_im_t = KL_batchmean(F.log_softmax(pseu_label, -1), F.softmax(logits_t.detach(), dim=-1))

                    kl_loss_s_s_im = KL_batchmean(F.log_softmax(logits_s, -1), F.softmax(outputs_.detach(), dim=-1))
                    kl_loss_s_im_s = KL_batchmean(F.log_softmax(outputs_, -1), F.softmax(logits_s.detach(), dim=-1))

                    kendall = np.empty(logits_t.size(0))
                    for j in range(logits_t.size(0)):
                        temp = \
                            kendalltau(logits_t[j, :].cpu().detach().numpy(), logits_s[j, :].cpu().detach().numpy())[0]
                        kendall[j] = temp
                    kendall = torch.from_numpy(kendall).to(device)
                    kendall_mean = kendall.mean()
                    mask = torch.where(kendall > args.krc, 1, 0)
                    mask_num = kendall.sum()
                    del kendall

                    if mask.sum() == 0:
                        kl_loss_st_s_t = 0
                        kl_loss_st_t_s = 0
                    else:
                        kl_loss_st_s_t = ntkl(logits_s, logits_t.detach(), labels, mask, KL_none)
                        kl_loss_st_t_s = ntkl(logits_t, logits_s.detach(), labels, mask, KL_none)

                    hooks_remover(hooks_t)
                    hooks_remover(hooks_s)

                    loss = (loss_t + loss_s +
                    kl_loss_t_t_im + kl_loss_t_im_t +
                    kl_loss_s_s_im + kl_loss_s_im_s +
                    kl_loss_st_s_t + kl_loss_st_t_s)
                    L_t = L_t + loss.item()
                    metric.update(outputs_s, label)
                L_t = L_t / (i + 1)
            res = metric.compute()
            print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_t:.4f}, OA = {float(res['Accuracy'])}")
            writer.add_scalar('test/Loss', L_t, epoch)
            writer.add_scalar('test/Acc', float(res['Accuracy']), epoch)

            # # For NYU-Depth-V2
            # print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_t:.4f}")
            # writer.add_scalar('test/Loss', L_t, epoch)

            # if (epoch + 1) % 10 == 0:
            #     print('\n===============Metrics==================')
            #     for e in res.keys():
            #         print(e)
            #         print(res[e])
            #         print('----------------------------')
            #     print('=======================================\n')

            # scheduler_t.step()  # 学习率衰减(当训练SAFN模型时，需加入监听指标作为参数)

        args.lr = max(1e-4, args.lr * ((args.epochs - epoch - 1) / (args.epochs - epoch)) ** 0.9)
        start_time2 = time.time()
        time_cost = start_time2 - start_time1
        if time_cost > 100:
            print(f"Epoch {epoch + 1} time cost: {time_cost / 60:.2f} minutes.\n")
        else:
            print(f"Epoch {epoch + 1} time cost: {time_cost:.2f} seconds.\n")

    writer.close()

    os.makedirs('./checkpoints/stu/c2kd', exist_ok=True)
    if args.save_model:
        torch.save(best_model_state,
                   f'./checkpoints/stu/c2kd/{args.database}_{args.Tmodel}_ORG--{args.Smodel}_seed{args.seed}_{args.mode}_ep{best_epoch + 1}-{args.epochs}.pth')

