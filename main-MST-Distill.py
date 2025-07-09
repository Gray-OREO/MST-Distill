import torch
from torch.utils.data import DataLoader
from torch import nn
from utils import get_Tmodules, get_AUXmodules, get_Smodules, get_data, get_dataset, seed_all
import copy
import torch.optim as optim
from KD_methods.MST import (get_MSTmodules, distillation_loss, distillation_loss_detached, hook_builder, TGroup_forward, hooks_remover,
                            AGroup_forward, load_balancing_loss, k_ways_MTGdistillation_detached)
import time
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Dataset import MultiModalX
import sys
import os


if __name__ == '__main__':
    '''
    Args Setting for CML.
    '''
    # EXP Setting
    parser = ArgumentParser(description='MST-Distill')
    parser.add_argument('--database', type=str, default='AV-MNIST',
                        help="database name must be one of "
                             "['NYU-Depth-V2', 'RAVDESS', 'AV-MNIST', 'VGGSound-50k', 'MM-IMDb']")
    parser.add_argument('--Tmodel', type=str, default='CNN-I',
                        help='Teacher model name')
    parser.add_argument('--AUXmodel', type=str, default='ThreeLayerCNN-A',
                        help='Aux model name')
    parser.add_argument('--Smodel', type=str, default='LeNet5',
                        help='Student model name')
    parser.add_argument('--mode', type=str, default='m1',
                        help='Target modality mode for Stu. model: m1 or m2')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers for DataLoader')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='cuda id')

    # Hyper-parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for training')
    parser.add_argument('--mask_layer_num', type=int, default=2,
                        help='num of MM-teacher models')
    parser.add_argument('--mask_head_num', type=int, default=3,
                        help='num of MM-teacher models')
    parser.add_argument('--k', type=int, default=1,
                        help='Top-k ways for MTGdistillation')
    parser.add_argument('--lb_mode', type=str, default='KL',
                        help='Load balancing mode: KL or CV')
    parser.add_argument('--w_task', type=float, default=1.,
                        help='weight for Stu. model loss_task')
    parser.add_argument('--w_kl', type=float, default=1.,
                        help='weight for Stu. model loss_Top-K_kl')
    parser.add_argument('--w_lb', type=float, default=1.,
                        help='weight for Stu. model loss_lb')
    parser.add_argument('--epochs_s1', type=int, default=100,
                        help='epochs for training')
    parser.add_argument('--epochs_s2', type=int, default=100,
                        help='epochs for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs for training')
    parser.add_argument('--s1_lr', type=float, default=0.0001,
                        help='learning rate for training')
    parser.add_argument('--s2_lr', type=float, default=0.0001,
                        help='learning rate for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for training')

    # Others
    parser.add_argument('--record', type=bool, default=True,
                        help='flag whether to record the learning log')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='flag whether to save best model')
    parser.add_argument('--test_phase', type=bool, default=False,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--preoccupied', action='store_true',
                        help='flag whether to preoccupy the GPU')
    parser.add_argument('--final_test', type=bool, default=False,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--commit', type=str, default='MST-Distill',
                        help='Commit for logs')
    args = parser.parse_args()

    seed_all(args.seed)

    # 保存log
    log_dir = f'./logs/mst'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = log_dir + (f'/{args.database}_{str(time.time()).split(".")[0]}_{args.commit}')
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
    model_t, _, _, criterion_t, preprocessing_t, postprocessing_t, _ = get_Tmodules(args, device)
    model_a, _, _, criterion_a, preprocessing_a, postprocessing_a, _ = get_AUXmodules(args)
    model_s, _, _, criterion_s, preprocessing_s, postprocessing_s, metric_s = get_Smodules(args)
    T_feat_names, A_feat_names, MaskNets_T, MaskNets_A, GateNet = get_MSTmodules(args)

    optimizer_s1 = optim.Adam(list(model_t.parameters()) + list(model_s.parameters()) + list(model_a.parameters()), lr=args.s1_lr)
    optimizers_MNt_s2 = [optim.Adam(MaskNet.parameters(), lr=args.s2_lr) for MaskNet in MaskNets_T]
    optimizers_MNa_s2 = [optim.Adam(MaskNet.parameters(), lr=args.s2_lr) for MaskNet in MaskNets_A]

    all_params = list(model_s.parameters()) + list(GateNet.parameters())
    optimizers_s3 = optim.Adam(all_params, lr=args.lr)

    model_t = model_t.to(device)
    model_a = model_a.to(device)
    model_s = model_s.to(device)
    GateNet.to(device)
    for MaskNet in MaskNets_T:
        MaskNet.to(device)
    for MaskNet in MaskNets_A:
        MaskNet.to(device)

    best_model_state = None
    best_Tmn_states = None
    best_Amn_states = None
    best_val_loss = float('inf')
    best_mn_val_loss = float('inf')

    if args.preoccupied:
        total_mem = torch.cuda.get_device_properties(device).total_memory
        reserverd_mem = int(total_mem * 0.5)
        dummy = torch.zeros(reserverd_mem, dtype=torch.uint8, device=device)

    print('================= Stage 1: Task-Oriented Co-Training =================')
    for epoch in range(args.epochs_s1):
        start_time1 = time.time()

        model_s.train()
        model_t.train()
        model_a.train()
        Loss_s1_all, Loss_t, Loss_s, Loss_a = 0, 0, 0, 0
        for i, (data, data2, label) in tqdm(enumerate(train_loader), desc="Stage 1: Model Co-training...",
                                            total=len(train_loader), dynamic_ncols=True, disable=False,
                                            file=sys.stdout):
            data, data2, label = data.to(device), data2.to(device), label.to(device)
            data_s = data if args.mode == 'm1' else data2
            data_a = data2 if args.mode == 'm1' else data

            outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
            outputs_a = model_a(data_a) if preprocessing_a is None else preprocessing_a(model_a, data_a)
            outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

            task_loss_t = criterion_t(outputs_t, label) if postprocessing_t is None else postprocessing_t(outputs_t, label)
            task_loss_a = criterion_a(outputs_a, label) if postprocessing_a is None else postprocessing_a(outputs_a, label)
            task_loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)

            kl_loss_ts = distillation_loss(args, outputs_s, outputs_t)
            kl_loss_as = distillation_loss(args, outputs_s, outputs_a)

            kl_loss_ta = distillation_loss(args, outputs_a, outputs_t)
            kl_loss_sa = distillation_loss(args, outputs_a, outputs_s)

            kl_loss_st = distillation_loss(args, outputs_t, outputs_s)
            kl_loss_at = distillation_loss(args, outputs_t, outputs_a)

            loss_t = task_loss_t + kl_loss_st + kl_loss_at
            loss_s = task_loss_s + kl_loss_ts + kl_loss_as
            loss_a = task_loss_a + kl_loss_ta + kl_loss_sa

            loss = loss_t + loss_s + loss_a

            optimizer_s1.zero_grad()
            loss.backward()
            optimizer_s1.step()

            Loss_a += loss_a.item()
            Loss_t += loss_t.item()
            Loss_s += loss_s.item()
            Loss_s1_all += loss.item()

        Loss_a_avg = Loss_a / (i + 1)
        Loss_t_avg = Loss_t / (i + 1)
        Loss_s_avg = Loss_s / (i + 1)
        Loss_s1_all_avg = Loss_s1_all / (i + 1)

        print(f'Train =====> Epoch {epoch + 1}/{args.epochs_s1}: loss_s1_all = {Loss_s1_all_avg:.4f} | '
              f'loss_t = {Loss_t_avg:.4f} | loss_a = {Loss_a_avg:.4f} | '
              f'loss_s = {Loss_s_avg:.4f}')
        writer.add_scalar('S1_CoTraining/Loss_s1_all', Loss_s1_all_avg, epoch)
        writer.add_scalar('S1_CoTraining/Loss_t', Loss_t_avg, epoch)
        writer.add_scalar('S1_CoTraining/Loss_a', Loss_a_avg, epoch)
        writer.add_scalar('S1_CoTraining/Loss_s', Loss_s_avg, epoch)

        start_time2 = time.time()
        time_cost = start_time2 - start_time1
        if time_cost > 100:
            print(f"Epoch {epoch + 1} time cost: {time_cost / 60:.2f} minutes.\n")
        else:
            print(f"Epoch {epoch + 1} time cost: {time_cost:.2f} seconds.\n")


    print('================= Stage 2: Dedicated Teacher Training =================')
    for epoch in range(args.epochs_s2):
        start_time1 = time.time()
        # training phase
        model_s.eval()
        model_t.eval()
        model_a.eval()
        for MaskNet in MaskNets_T:
            MaskNet.train()
        for MaskNet in MaskNets_A:
            MaskNet.train()
        Loss_s2_all, Loss_MNt, Loss_MNa = 0, 0, 0
        for i, (data, data2, label) in tqdm(enumerate(train_loader), desc="Stage 2: MaskNets Training...",
                                            total=len(train_loader), dynamic_ncols=True, disable=False,
                                            file=sys.stdout):
            data, data2, label = data.to(device), data2.to(device), label.to(device)
            data_s = data if args.mode == 'm1' else data2
            data_a = data2 if args.mode == 'm1' else data

            outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

            loss_tmp_t = 0
            for j, (MaskNet, feat_name, opt) in enumerate(zip(MaskNets_T, T_feat_names, optimizers_MNt_s2)):
                hook_t, features_t = hook_builder(model_t, feat_name, MaskNet)
                outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
                loss_kl = distillation_loss_detached(args, outputs_t, outputs_s)
                opt.zero_grad()
                loss_kl.backward()
                opt.step()
                hook_t.remove()
                loss_tmp_t += loss_kl.item()
            loss_tmp_t = loss_tmp_t / len(MaskNets_T)

            loss_tmp_a = 0
            for j, (MaskNet, feat_name, opt) in enumerate(zip(MaskNets_A, A_feat_names, optimizers_MNa_s2)):
                hook_a, features_a = hook_builder(model_a, feat_name, MaskNet)
                outputs_a = model_a(data_a) if preprocessing_a is None else preprocessing_a(model_a, data_a)
                loss_kl = distillation_loss_detached(args, outputs_a, outputs_s)
                opt.zero_grad()
                loss_kl.backward()
                opt.step()
                hook_a.remove()
                loss_tmp_a += loss_kl.item()
            loss_tmp_a = loss_tmp_a / len(MaskNets_A)

            Loss_s2_all += (loss_tmp_t + loss_tmp_a)
            Loss_MNt += loss_tmp_t
            Loss_MNa += loss_tmp_a

        Loss_MNt_avg = Loss_MNt / (i + 1)
        Loss_MNa_avg = Loss_MNa / (i + 1)
        Loss_s2_all_avg = Loss_s2_all / (i + 1)

        print(f'Train =====> Epoch {epoch + 1}/{args.epochs_s2}: loss_s2_all = {Loss_s2_all_avg:.4f} | '
              f'loss_MNt = {Loss_MNt_avg:.4f} | loss_MNa = {Loss_MNa_avg:.4f}')
        writer.add_scalar('S2_MNTraining/Loss_s2_all', Loss_s2_all_avg, epoch)
        writer.add_scalar('S2_MNTraining/Loss_MNt', Loss_MNt_avg, epoch)
        writer.add_scalar('S2_MNTraining/Loss_MNa', Loss_MNa_avg, epoch)

        # validation phase
        model_s.eval()
        model_t.eval()
        model_a.eval()
        for MaskNet in MaskNets_T:
            MaskNet.eval()
        for MaskNet in MaskNets_A:
            MaskNet.eval()
        with torch.no_grad():
            Loss_s2_all, Loss_MNt, Loss_MNa = 0, 0, 0
            for i, (data, data2, label) in tqdm(enumerate(valid_loader), desc="MaskNet Validating ...",
                                                total=len(valid_loader), dynamic_ncols=True, disable=False,
                                                file=sys.stdout):
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                data_s = data if args.mode == 'm1' else data2
                data_a = data2 if args.mode == 'm1' else data
                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                loss_tmp_t = 0
                for j, (MaskNet, feat_name, opt) in enumerate(zip(MaskNets_T, T_feat_names, optimizers_MNt_s2)):
                    hook_t, features_t = hook_builder(model_t, feat_name, MaskNet)
                    outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
                    loss_kl = distillation_loss_detached(args, outputs_t, outputs_s)
                    hook_t.remove()
                    loss_tmp_t += loss_kl.item()
                loss_tmp_t = loss_tmp_t / len(MaskNets_T)

                loss_tmp_a = 0
                for j, (MaskNet, feat_name, opt) in enumerate(zip(MaskNets_A, A_feat_names, optimizers_MNa_s2)):
                    hook_a, features_a = hook_builder(model_a, feat_name, MaskNet)
                    outputs_a = model_a(data_a) if preprocessing_a is None else preprocessing_a(model_a, data_a)
                    loss_kl = distillation_loss_detached(args, outputs_a, outputs_s)
                    hook_a.remove()
                    loss_tmp_a += loss_kl.item()
                loss_tmp_a = loss_tmp_a / len(MaskNets_A)

                Loss_s2_all += (loss_tmp_t + loss_tmp_a)
                Loss_MNt += loss_tmp_t
                Loss_MNa += loss_tmp_a

            Loss_MNt_avg = Loss_MNt / (i + 1)
            Loss_MNa_avg = Loss_MNa / (i + 1)
            Loss_s2_all_avg = Loss_s2_all / (i + 1)

            print(f'Valid =====> Epoch {epoch + 1}/{args.epochs_s2}: loss_s2_all = {Loss_s2_all_avg:.4f} | '
                  f'loss_MNt = {Loss_MNt_avg:.4f} | loss_MNa = {Loss_MNa_avg:.4f}')
            writer.add_scalar('S2_MNValid/Loss_s2_all', Loss_s2_all_avg, epoch)
            writer.add_scalar('S2_MNValid/Loss_MNt', Loss_MNt_avg, epoch)
            writer.add_scalar('S2_MNValid/Loss_MNa', Loss_MNa_avg, epoch)


            if Loss_s2_all_avg < best_mn_val_loss:
                best_mn_val_loss = Loss_s2_all_avg
                best_Tmn_states = [copy.deepcopy(MaskNet.state_dict()) for MaskNet in MaskNets_T]
                best_Amn_states = [copy.deepcopy(MaskNet.state_dict()) for MaskNet in MaskNets_A]

        start_time2 = time.time()
        time_cost = start_time2 - start_time1
        if time_cost > 100:
            print(f"Epoch {epoch + 1} time cost: {time_cost / 60:.2f} minutes.\n")
        else:
            print(f"Epoch {epoch + 1} time cost: {time_cost:.2f} seconds.\n")


    print('================= Stage 3: Mixture of Teacher Distillation =================')
    if args.preoccupied:
        del dummy
        torch.cuda.empty_cache()

    for MaskNet, weights in zip(MaskNets_T, best_Tmn_states):
        MaskNet.load_state_dict(weights)
    for MaskNet, weights in zip(MaskNets_A, best_Amn_states):
        MaskNet.load_state_dict(weights)

    for MaskNet in MaskNets_T:
        MaskNet.eval()
    for MaskNet in MaskNets_A:
        MaskNet.eval()
    model_t.eval()
    model_a.eval()

    for epoch in range(args.epochs):
        start_time1 = time.time()
        model_s.train()
        GateNet.train()
        Loss_s_all, task_Loss_s, kl_Loss_s, lb_Loss_s = 0, 0, 0, 0
        task_Loss_DTG = 0
        routing_count = [0 for _ in range(len(MaskNets_T) + len(MaskNets_A))]
        for i, (data, data2, label) in tqdm(enumerate(train_loader), desc="Stage 3: Dedicated Teacher Group Distillation...",
                                            total=len(train_loader), dynamic_ncols=True, disable=False,
                                            file=sys.stdout):
            data, data2, label = data.to(device), data2.to(device), label.to(device)
            data_s = data if args.mode == 'm1' else data2
            data_a = data2 if args.mode == 'm1' else data

            outputs_t_list, hooks_t = TGroup_forward(data, data2, model_t, MaskNets_T, T_feat_names, preprocessing_t)
            outputs_a_list, hooks_a = AGroup_forward(data_a, model_a, MaskNets_A, A_feat_names, preprocessing_a)
            outputs_DTG = outputs_t_list + outputs_a_list
            outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

            gate_logits = GateNet(outputs_s)
            gate_logits_softmax = nn.Softmax(dim=1)(gate_logits)

            task_loss_t_list = [criterion_t(outputs_t, label) if postprocessing_t is None else postprocessing_t(outputs_t, label)
                                for outputs_t in outputs_t_list]
            task_loss_a_list = [criterion_a(outputs_a, label) if postprocessing_a is None else postprocessing_a(outputs_a, label)
                                for outputs_a in outputs_a_list]
            task_loss_DTG = task_loss_t_list + task_loss_a_list
            task_loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)

            loss_lb = load_balancing_loss(gate_logits_softmax, mode=args.lb_mode)
            TopK_DTD_loss, TopK_indices = k_ways_MTGdistillation_detached(args, outputs_DTG, outputs_s, gate_logits_softmax, args.k)

            loss = args.w_task * task_loss_s + args.w_kl * TopK_DTD_loss + args.w_lb * loss_lb

            optimizers_s3.zero_grad()
            loss.backward()
            optimizers_s3.step()
            hooks_remover(hooks_t)
            hooks_remover(hooks_a)

            task_Loss_DTG_ = torch.mean(torch.stack(task_loss_DTG)).item()
            task_Loss_DTG += task_Loss_DTG_
            task_Loss_s += task_loss_s.item()
            kl_Loss_s += TopK_DTD_loss.item()
            lb_Loss_s += loss_lb.item()
            Loss_s_all += loss.item()
            count = torch.bincount(TopK_indices.view(-1), minlength=len(MaskNets_T) + len(MaskNets_A))
            routing_count = [c + c_new for c, c_new in zip(routing_count, count)]

        task_Loss_DTG_avg = task_Loss_DTG / (i + 1)
        task_Loss_s_avg = task_Loss_s / (i + 1)
        kl_Loss_s_avg = kl_Loss_s / (i + 1)
        lb_Loss_s_avg = lb_Loss_s / (i + 1)
        Loss_s_all_avg = Loss_s_all / (i + 1)
        routing_count_avg = [c / (i + 1) for c in routing_count]
        # routing probability
        routing_count_sum = sum(routing_count_avg)
        routing_count_avg = [c / routing_count_sum for c in routing_count_avg]

        print(f'Train =====> Epoch {epoch + 1}/{args.epochs}: loss_s_all = {Loss_s_all_avg:.4f} | '
              f'loss_s_Top-k = {kl_Loss_s_avg:.4f} | loss_s_task = {task_Loss_s_avg:.4f} | '
              f'loss_s_lb = {lb_Loss_s_avg:.4f} | loss_DTG_task = {task_Loss_DTG_avg:.4f}')
        writer.add_scalar('S3_DTGD/Loss_s_all', Loss_s_all_avg, epoch)
        writer.add_scalar('S3_DTGD/Loss_s_Top-k', kl_Loss_s_avg, epoch)
        writer.add_scalar('S3_DTGD/Loss_s_task', task_Loss_s_avg, epoch)
        writer.add_scalar('S3_DTGD/Loss_s_lb', lb_Loss_s_avg, epoch)
        writer.add_scalar('S3_DTGD/Loss_DTG_task', task_Loss_DTG_avg, epoch)
        for id in range(len(MaskNets_T) + len(MaskNets_A)):
            writer.add_scalar(f'S3_DTGD_Routing/DT_{id}', routing_count_avg[id], epoch)

        # Validation ---------------------------------------------------------
        metric_s.reset()
        L_s_val = 0
        acc_c = 0
        Loss_S = 0
        gt_list, pred_list = [], []
        model_s.eval()
        GateNet.eval()
        with torch.no_grad():
            for i, (data, data2, label) in tqdm(enumerate(valid_loader), desc="UniM-Stu. Model Validating ...",
                                                total=len(valid_loader), dynamic_ncols=True, disable=False,
                                                file=sys.stdout):
                # print('Val Iter {}'.format(i))
                data, data2, label = data.to(device), data2.to(device), label.to(device)

                data_s = data if args.mode == 'm1' else data2

                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)
                loss = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
                L_s_val += loss.item()
                metric_s.update(outputs_s, label)
            L_s_val = L_s_val / (i + 1)

        # res = metric.compute()
        # print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_s_val:.4f}, OA_s = {res['Accuracy']} | "
        #       f"loss_S = {Loss_S_avg:.4f} | "
        #       f"loss_recon_s = {Loss_recon_s_avg:.4f} | loss_kd = {Loss_kd_avg:.4f}")
        # writer.add_scalar('valid/Loss', L_s_val, epoch)
        # writer.add_scalar('valid/Acc', float(res['Accuracy']), epoch)

        # Brief Summary
        print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_s_val:.4f}")
        writer.add_scalar('valid/Loss', L_s_val, epoch)


        if L_s_val < best_val_loss:
            best_val_loss = L_s_val
            best_model_state = copy.deepcopy(model_s.state_dict())
            best_epoch = epoch

        # test
        if args.test_phase:
            model_s.load_state_dict(best_model_state)
            model_s.eval()
            metric_s.reset()
            L_t = 0
            acc_c = 0
            Loss_S = 0
            gt_list, pred_list = [], []
            with torch.no_grad():
                for i, (data, data2, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                                    total=len(test_loader), dynamic_ncols=True, disable=False,
                                                    file=sys.stdout):
                    data, data2, label = data.to(device), data2.to(device), label.to(device)

                    data_s = data if args.mode == 'm1' else data2

                    outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                    loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(
                        outputs_s, label)

                    loss = loss_s
                    Loss_S += loss_s.item()

                    metric_s.update(outputs_s, label)
                Loss_S_avg = Loss_S / (i + 1)

            # res = metric.compute()
            # print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss_s = {Loss_S_avg:.4f}")
            # writer.add_scalar('test/Loss', L_t, epoch)
            # writer.add_scalar('test/Acc', float(res['Accuracy']), epoch)

            # Brief Summary
            print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss_s = {Loss_S_avg:.4f}")
            writer.add_scalar('test/Loss', Loss_S_avg, epoch)

            # if (epoch + 1) % 10 == 0:
            #     print('\n===============Metrics==================')
            #     for e in res.keys():
            #         print(e)
            #         print(res[e])
            #         print('----------------------------')
            #     print('=======================================\n')

            # scheduler_t.step()  # 学习率衰减(当训练SAFN模型时，需加入监听指标作为参数)
        args.w_lb *= 0.9 if epoch % 10 == 0 else 1
        args.w_kl *= 0.5 if epoch % 30 == 0 else 1
        start_time2 = time.time()
        time_cost = start_time2 - start_time1
        if time_cost > 100:
            print(f"Epoch {epoch + 1} time cost: {time_cost / 60:.2f} minutes.\n")
        else:
            print(f"Epoch {epoch + 1} time cost: {time_cost:.2f} seconds.\n")

    writer.close()

    if args.final_test:
        model_s.load_state_dict(best_model_state)
        print('================= Final Test for This Model =================')
        # test
        model_s.eval()
        metric_s.reset()
        gt_list, pred_list = [], []
        with torch.no_grad():
            for i, (data, data2, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                         total=len(test_loader), dynamic_ncols=True, disable=True, file=sys.stdout):
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                data_s = data if args.mode == 'm1' else data2
                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)
                metric_s.update(outputs_s, label)
        res = metric_s.compute()

        print('\n===============Metrics==================')
        for e in res.keys():
            print(e)
            print(res[e])
            print('----------------------------')
        print('=======================================\n')

    if args.save_model:
        os.makedirs('./checkpoints/stu/mst', exist_ok=True)
        torch.save(best_model_state,
                   f'./checkpoints/stu/mst/{args.database}_{args.Tmodel}--{args.Smodel}_seed{args.seed}_{args.mode}_ep{best_epoch + 1}-{args.epochs}.pth')