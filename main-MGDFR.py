import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import os
from utils import seed_all, get_data, get_dataset, get_Tmodules, get_Smodules, min_max_normalize
from KD_methods.MGDFR import hooks_builder, hooks_remover
import copy
import numpy as np
import torch.optim as optim
from KD_methods.MGDFR import get_MGDFRmodules
from KD_methods.KD import distillation_loss
import time
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Dataset import MultiModalX
import sys


if __name__ == '__main__':
    '''
    Args Setting for CML.
    '''
    parser = ArgumentParser(description='CML-MGDFR')
    parser.add_argument('--database', type=str, default='RAVDESS',
                        help="database name must be one of "
                             "['NYU-Depth-V2', 'RAVDESS', 'AV-MNIST', 'VGGSound-50k', 'MM-IMDb']")
    parser.add_argument('--Tmodel', type=str, default='VisualBranchNet',
                        help='Teacher model name')

    parser.add_argument('--Smodel', type=str, default='AudioBranchNet',
                        help='Student model name')
    parser.add_argument('--mode', type=str, default='m2',
                        help='Data mode: m1 or m2')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers for DataLoader')
    parser.add_argument('--ratio', type=float, default=0.75,
                        help='remove feature dimension ratio')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight for KL loss')
    parser.add_argument('--co_lr', type=float, default=0.001,
                        help='learning rate for co-training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for training')
    parser.add_argument('--record', type=bool, default=True,
                        help='flag whether to record the learning log')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='cuda id')
    parser.add_argument('--co_epochs', type=int, default=100,
                        help='epochs for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs for training')
    parser.add_argument('--repeat_permute', type=int, default=10,
                        help='repeat number for permute')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='flag whether to save best model')
    parser.add_argument('--test_phase', type=bool, default=False,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--commit', type=str, default='MGDFR-baseline',
                        help='Commit for logs')
    args = parser.parse_args()

    seed_all(args.seed)


    log_dir = f'./logs/mgdfr'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = log_dir + f'/{args.database}_{args.lr}_{str(time.time()).split(".")[0]}_{args.commit}'
    writer = SummaryWriter(log_dir, write_to_disk=args.record)

    data = get_data(args.database)
    data_train = get_dataset(args.database, data, 'train', args.seed)
    data_val = get_dataset(args.database, data, 'val', args.seed)
    data_test = get_dataset(args.database, data, 'test', args.seed)

    train_dataset = MultiModalX(data_train, args.database, 'ORG')
    valid_dataset = MultiModalX(data_val, args.database, 'ORG')
    test_dataset = MultiModalX(data_test, args.database, 'ORG')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        shuffle=True
    )

    calcu_loader = DataLoader(
        train_dataset,
        batch_size=6 * args.batch_size,
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
    model_t, _, scheduler_t, criterion_t, preprocessing_t, postprocessing_t, metric_t = get_Tmodules(args, device)
    model_s, optimizer_s, scheduler_s, criterion_s, preprocessing_s, postprocessing_s, metric_s = get_Smodules(args)
    feat_name, feat_dim, salience_vector, criterion_dist = get_MGDFRmodules(args)
    optimizer = torch.optim.SGD(list(model_t.parameters()) + list(model_s.parameters()), lr=args.co_lr, momentum=0.9)

    model_t = model_t.to(device)
    model_s = model_s.to(device)
    salience_vector = salience_vector.to(device)


    os.makedirs('./checkpoints/stu/mgdfr', exist_ok=True)
    salience_vector_path = './checkpoints/stu/mgdfr/salience_vector'
    os.makedirs(salience_vector_path, exist_ok=True)

    salience_vector_name = f'{args.database}_{args.Tmodel}_{args.Smodel}_seed{args.seed}_{args.mode}_co-ep{args.co_epochs}_sv.pth'

    os.makedirs('./checkpoints/stu/mgdfr/overlap_tea', exist_ok=True)
    overlap_tea_model_path = (f'./checkpoints/stu/mgdfr/overlap_tea/{args.database}_{args.Tmodel}-'
                              f'-{args.Smodel}_seed{args.seed}_{args.mode}_co-ep{args.co_epochs}_overlap_tea.pth')
    os.makedirs('./checkpoints/stu/mgdfr/overlap_stu', exist_ok=True)
    overlap_stu_model_path = (f'./checkpoints/stu/mgdfr/overlap_stu/{args.database}_{args.Tmodel}-'
                              f'-{args.Smodel}_seed{args.seed}_{args.mode}_co-ep{args.co_epochs}_overlap_stu.pth')

    Flag_joint_learning = os.path.exists(overlap_tea_model_path)
    Flag_sv_calcu = os.path.exists(f'{salience_vector_path}/{salience_vector_name}')

    if not Flag_joint_learning:
        # Tea. & Stu. model joint learning
        print('================= Tea. & Stu. Model Joint Learning =================')
        for epoch in range(args.co_epochs):
            start_time1 = time.time()
            # train
            model_t.train()
            model_s.train()
            if args.Tmodel == 'CEN':
                for module in model_t.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            if args.Smodel in ['CEN_RGB-branch', 'CEN_D-branch']:
                for module in model_s.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

            tra_LOSS_t, tra_LOSS_s, tra_LOSS_dist = 0, 0, 0
            for i, (data, data2, label) in tqdm(enumerate(train_loader), desc="T&S Model Co-Training ...",
                                                total=len(train_loader), dynamic_ncols=True,
                                                disable=False, file=sys.stdout):
                # print('Train Iter {}'.format(i))
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                if args.mode == 'm1':
                    data_t, data_s = data2, data
                else:
                    data_t, data_s = data, data2
                outputs_t = model_t(data_t) if preprocessing_t is None else preprocessing_t(model_t, data_t)
                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                loss_t = criterion_t(outputs_t, label) if postprocessing_t is None else postprocessing_t(outputs_t, label)
                loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
                loss_dist = criterion_dist(outputs_t, outputs_s)
                loss = loss_t + loss_s + loss_dist

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tra_LOSS_t += loss_t.item()
                tra_LOSS_s += loss_s.item()
                tra_LOSS_dist += loss_dist.item()

            tra_LOSS_t_avg = tra_LOSS_t / (i + 1)
            tra_LOSS_s_avg = tra_LOSS_s / (i + 1)
            tra_LOSS_dist_avg = tra_LOSS_dist / (i + 1)
            print(f'Train =====> Epoch {epoch + 1}/{args.co_epochs}: loss_t = {tra_LOSS_t_avg:.4f}  |  loss_s = {tra_LOSS_s_avg:.4f}  |  '
                  f'loss_dist = {tra_LOSS_dist_avg:.4f}')
            # writer.add_scalar('Co-train/Loss_t', tra_LOSS_t_avg, epoch)
            # writer.add_scalar('Co-train/Loss_s', tra_LOSS_s_avg, epoch)
            # writer.add_scalar('Co-train/Loss_dist', tra_LOSS_dist_avg, epoch)

            start_time2 = time.time()
            time_cost = start_time2 - start_time1
            if time_cost > 100:
                print(f"Co-train Epoch {epoch + 1} time cost: {time_cost / 60:.2f} minutes.\n")
            else:
                print(f"Co-train Epoch {epoch + 1} time cost: {time_cost:.2f} seconds.\n")

        torch.save(model_t.state_dict(), overlap_tea_model_path)
        torch.save(model_s.state_dict(), overlap_stu_model_path)
        writer.close()

    if not Flag_sv_calcu:
        if Flag_joint_learning:
            model_t.load_state_dict(torch.load(overlap_tea_model_path, map_location=device, weights_only=False))
            model_s.load_state_dict(torch.load(overlap_stu_model_path, map_location=device, weights_only=False))
        print('================= Tea. Feature Salience Vector Calculate =================')
        with torch.no_grad():
            model_t.eval()
            model_s.eval()
            for i, (data, data2, label) in tqdm(enumerate(calcu_loader), desc="Calculating ...",
                                                total=len(calcu_loader), dynamic_ncols=True,
                                                disable=False, file=sys.stdout):
                # print('Train Iter {}'.format(i))
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                if args.mode == 'm1':
                    data_t, data_s = data2, data
                else:
                    data_t, data_s = data, data2
                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                for j in range(args.repeat_permute):
                    for index in range(feat_dim):
                        hooks_t, features_t = hooks_builder(model_t, feat_name, 'permute', [index])
                        outputs_t_permu = model_t(data_t) if preprocessing_t is None else preprocessing_t(model_t, data_t)
                        salience_vector[j, index] += criterion_dist(outputs_t_permu, outputs_s)/ args.repeat_permute
                        hooks_remover(hooks_t)

            salience_vector = np.array(((salience_vector / len(train_dataset)).mean(dim=0)).cpu())
            salience_vector = min_max_normalize(salience_vector)
            torch.save(salience_vector, f'{salience_vector_path}/{salience_vector_name}')

    # KD After MGDFR
    salience_vector = torch.load(f'{salience_vector_path}/{salience_vector_name}', weights_only=False)
    sort_idx = (salience_vector).argsort()
    remove_idx = sort_idx[0: int(args.ratio * feat_dim)]

    model_t.load_state_dict(torch.load(overlap_tea_model_path, map_location=device, weights_only=False))
    model_t.eval()
    model_s, optimizer_s, scheduler_s, criterion_s, preprocessing_s, postprocessing_s, metric = get_Smodules(args)
    model_s = model_s.to(device)

    best_model_state = None
    best_val_loss = float('inf')

    print('================= KD After MGDFR =================')
    for epoch in range(args.epochs):
        start_time1 = time.time()
        # train
        model_s.train()
        if args.Tmodel == 'CEN':
            for module in model_t.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        if args.Smodel in ['CEN_RGB-branch', 'CEN_D-branch']:
            for module in model_s.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
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

            hooks_t, features_t = hooks_builder(model_t, feat_name, 'freeze', remove_idx)

            outputs_t = model_t(data_t) if preprocessing_t is None else preprocessing_t(model_t, data_t)
            outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

            hooks_remover(hooks_t)

            loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
            loss_kl = distillation_loss(args, outputs_s, outputs_t)
            loss = (1.0 - args.alpha) * loss_s + args.alpha * loss_kl
            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()
            tra_LOSS_s += loss.item()

        tra_LOSS_s_avg = tra_LOSS_s / (i + 1)
        print(f'Train =====> Epoch {epoch + 1}/{args.epochs}: loss = {tra_LOSS_s_avg:.4f}')
        writer.add_scalar('train/Loss', tra_LOSS_s_avg, epoch)

        # validation
        model_s.eval()
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

                hooks_t, features_t = hooks_builder(model_t, feat_name, 'freeze', remove_idx)

                outputs_t = model_t(data_t) if preprocessing_t is None else preprocessing_t(model_t, data_t)
                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                hooks_remover(hooks_t)

                loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
                loss_kl = distillation_loss(args, outputs_s, outputs_t)
                loss = loss_s
                L_s_val += loss.item()
                metric.update(outputs_s, label)
            L_s_val = L_s_val / (i + 1)
        # res = metric.compute()
        # print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_s_val:.4f}, OA_s = {float(res['Accuracy'])}")
        # writer.add_scalar('valid/Loss', L_s_val, epoch)
        # writer.add_scalar('valid/Acc', float(res['Accuracy']), epoch)

        # For NYU-Depth-V2
        print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_s_val:.4f}")
        writer.add_scalar('valid/Loss', L_s_val, epoch)


        if L_s_val < best_val_loss:
            best_val_loss = L_s_val
            best_model_state = copy.deepcopy(model_s.state_dict())
            best_epoch = epoch

        # test
        if args.test_phase:
            model_s.eval()
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

                    hooks_t, features_t = hooks_builder(model_t, feat_name, 'freeze', remove_idx)

                    outputs_t = model_t(data_t) if preprocessing_t is None else preprocessing_t(model_t, data_t)
                    outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                    hooks_remover(hooks_t)

                    loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
                    loss_kl = distillation_loss(args, outputs_s, outputs_t)
                    loss = (1.0 - args.alpha) * loss_s + args.alpha * loss_kl
                    L_t = L_t + loss.item()
                    metric.update(outputs_s, label)
                L_t = L_t / (i + 1)
            res = metric.compute()
            print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_t:.4f}, OA_s = {float(res['Accuracy'])}")
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
        args.alpha *= 0.5 if (epoch + 1) % 30 == 0 else 1.0
        start_time2 = time.time()
        time_cost = start_time2 - start_time1
        if time_cost > 100:
            print(f"Epoch {epoch + 1} time cost: {time_cost / 60:.2f} minutes.\n")
        else:
            print(f"Epoch {epoch + 1} time cost: {time_cost:.2f} seconds.\n")

    writer.close()

    if args.save_model:
        torch.save(best_model_state,
                   f'./checkpoints/stu/mgdfr/{args.database}_{args.Tmodel}--{args.Smodel}_seed{args.seed}_{args.mode}_ep{best_epoch + 1}-{args.epochs}.pth')

