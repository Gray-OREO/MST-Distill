import torch
from torch.utils.data import DataLoader
from torch import nn
from utils import seed_all, hooks_builder, hooks_remover, get_data, get_dataset, get_Smodules, get_Tmodules
import copy
import os
import time
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Dataset import MultiModalX
from KD_methods.OFA import get_OFAmodules, projectors_train, projectors_eval
import sys


if __name__ == '__main__':
    '''
    Args Setting for CML.
    '''
    parser = ArgumentParser(description='CML-OFA')
    parser.add_argument('--database', type=str, default='AV-MNIST',
                        help="database name must be one of ['NYU-Depth-V2', 'RAVDESS', 'AV-MNIST', 'VGGSound-50k', 'MM-IMDb']")
    parser.add_argument('--Tmodel', type=str, default='CNN-I',
                        help='Teacher model name')
    parser.add_argument('--Smodel', type=str, default='LeNet5',
                        help='Student model name')
    parser.add_argument('--mode', type=str, default='m1',
                        help='modality mode: m1 or m2')

    parser.add_argument('--ckpt_name', type=str,
                        default='AV-MNIST_CNN-I_seed0_ORG_ep5-5.pth',
                        help='The name of the weight to be loaded in ./checkpoints/stu')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers for DataLoader')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight for loss')
    parser.add_argument('--ofa_eps', type=float, default=1.1,
                        help='Modulating parameter')
    parser.add_argument('--ofa_temperature', type=float, default=1.,
                        help='The temperature fot OFA loss')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for training')
    parser.add_argument('--record', type=bool, default=True,
                        help='flag whether to record the learning log')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='cuda id')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs for training')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='flag whether to save best model')
    parser.add_argument('--test_phase', type=bool, default=False,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--commit', type=str, default='CMKD-OFA',
                        help='Commit for logs')
    args = parser.parse_args()

    seed_all(args.seed)


    log_dir = f'./logs/ofa'
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
    model_t, _, _, _, preprocessing_t, postprocessing_t, _ = get_Tmodules(args, device)
    model_s, _, scheduler_s, criterion_s, preprocessing_s, postprocessing_s, metric = get_Smodules(args)
    projectors, criterion_ofa, optimizer_ofa = get_OFAmodules(args, model_s)

    model_t = model_t.to(device)
    model_s = model_s.to(device)
    projectors = [projector.to(device) for projector in projectors]
    model_t.load_state_dict(torch.load(f'checkpoints/tea/{args.ckpt_name}', map_location=f'cuda:{args.cuda_id}', weights_only=True))
    model_t.eval()

    best_model_state = None
    best_val_loss = float('inf')
    # Cloud-only Learning
    print('================= CMKD-OFA =================')
    for epoch in range(args.epochs):
        start_time1 = time.time()
        # train
        model_s.train()
        projectors_train(projectors)
        if args.Tmodel == 'CEN':
            for module in model_t.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        if args.Smodel in ['CEN_RGB-branch', 'CEN_D-branch']:
            for module in model_s.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        tra_LOSS_s, tra_LOSS_ofa = 0, 0
        for i, (data, data2, label) in tqdm(enumerate(train_loader), desc="Model Training ...",
                                            total=len(train_loader), dynamic_ncols=True,
                                            disable=False, file=sys.stdout):
            # print('Train Iter {}'.format(i))
            data, data2, label = data.to(device), data2.to(device), label.to(device)
            data_s = data if args.mode == 'm1' else data2

            hooks_t, features_t = hooks_builder(model_t, model_t.hook_names)
            hooks_s, features_s = hooks_builder(model_s, model_s.hook_names)

            outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
            outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

            loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
            loss_ofa = criterion_ofa(features_t, features_s, label)

            hooks_remover(hooks_t)
            hooks_remover(hooks_s)

            loss = (1.0 - args.alpha) * loss_s + args.alpha * loss_ofa
            optimizer_ofa.zero_grad()
            loss.backward()
            optimizer_ofa.step()
            tra_LOSS_s += loss.item()
            tra_LOSS_ofa += loss_ofa.item()

        tra_LOSS_s_avg = tra_LOSS_s / (i + 1)
        tra_LOSS_ofa_avg = tra_LOSS_ofa / (i + 1)
        print(f'Train =====> Epoch {epoch + 1}/{args.epochs}: loss = {tra_LOSS_s_avg:.4f} | loss_ofa = {tra_LOSS_ofa_avg:.4f}')
        writer.add_scalar('train/Loss', tra_LOSS_s_avg, epoch)

        # validation
        model_s.eval()
        projectors_eval(projectors)
        metric.reset()
        L_s_val = 0
        val_LOSS_ofa = 0
        acc_c = 0
        gt_list, pred_list = [], []
        with torch.no_grad():
            for i, (data, data2, label) in tqdm(enumerate(valid_loader), desc="Model Validating ...",
                                                total=len(valid_loader), dynamic_ncols=True, disable=False,
                                                file=sys.stdout):
                # print('Val Iter {}'.format(i))
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                data_s = data if args.mode == 'm1' else data2

                hooks_t, features_t = hooks_builder(model_t, model_t.hook_names)
                hooks_s, features_s = hooks_builder(model_s, model_s.hook_names)

                outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
                loss_ofa = criterion_ofa(features_t, features_s, label)

                hooks_remover(hooks_t)
                hooks_remover(hooks_s)

                loss = loss_s
                L_s_val += loss.item()
                val_LOSS_ofa += loss_ofa.item()
                metric.update(outputs_s, label)
            L_s_val = L_s_val / (i + 1)
            val_LOSS_ofa_avg = val_LOSS_ofa / (i + 1)
        # res = metric.compute()
        # print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss_c = {L_s_val:.4f}, OA_c = {res['Accuracy']:.2f}%")
        # writer.add_scalar('valid/Loss', L_s_val, epoch)
        # writer.add_scalar('valid/Acc', res['Accuracy'], epoch)

        # For NYU-Depth-V2
        print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_s_val:.4f} | loss_ofa = {val_LOSS_ofa_avg:.4f}")
        writer.add_scalar('valid/Loss', L_s_val, epoch)

        # 保存验证集上表现最好的模型
        if L_s_val < best_val_loss:
            best_val_loss = L_s_val
            best_model_state = copy.deepcopy(model_s.state_dict())
            best_epoch = epoch

        # test
        if args.test_phase:
            model_s.eval()
            projectors_eval(projectors)
            metric.reset()
            L_t = 0
            acc_c = 0
            gt_list, pred_list = [], []
            with torch.no_grad():
                for i, (data, data2, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                                    total=len(test_loader), dynamic_ncols=True, disable=False,
                                                    file=sys.stdout):
                    data, data2, label = data.to(device), data2.to(device), label.to(device)
                    data_s = data if args.mode == 'm1' else data2

                    # hooks_t, features_t = hooks_builder(model_t, model_t.hook_names)
                    hooks_s, features_s = hooks_builder(model_s, model_s.hook_names)

                    # outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
                    outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)

                    loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
                    # loss_ofa = criterion_ofa(features_t, features_s, label)

                    # hooks_remover(hooks_t)
                    hooks_remover(hooks_s)

                    loss = loss_s
                    L_t = L_t + loss.item()
                    metric.update(outputs_s, label)
                L_t = L_t / (i + 1)
            # res = metric.compute()
            # print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss_c = {L_t:.4f}, OA_c = {res['Accuracy']:.2f}%")
            # writer.add_scalar('test/Loss', L_t, epoch)
            # writer.add_scalar('test/Acc', res['Accuracy'], epoch)

            # For NYU-Depth-V2
            print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_t:.4f}")
            writer.add_scalar('test/Loss', L_t, epoch)

            # if (epoch + 1) % 10 == 0:
            #     print('\n===============Metrics==================')
            #     for e in res.keys():
            #         print(e)
            #         print(res[e])
            #         print('----------------------------')
            #     print('=======================================\n')

            # scheduler_t.step()
        args.alpha *= 0.5 if (epoch + 1) % 30 == 0 else 1.0
        start_time2 = time.time()
        time_cost = start_time2 - start_time1
        if time_cost > 100:
            print(f"Epoch {epoch + 1} time cost: {time_cost / 60:.2f} minutes.\n")
        else:
            print(f"Epoch {epoch + 1} time cost: {time_cost:.2f} seconds.\n")

    writer.close()

    if args.save_model:
        if not os.path.exists('./checkpoints/stu/ofa'):
            os.makedirs('./checkpoints/stu/ofa')
        names = args.ckpt_name.split('_')
        Tmodel_mode = names[3]
        torch.save(best_model_state,
                   f'./checkpoints/stu/ofa/{args.database}_{args.Tmodel}_{Tmodel_mode}--{args.Smodel}_seed{args.seed}_{args.mode}_ep{best_epoch + 1}-{args.epochs}.pth')
