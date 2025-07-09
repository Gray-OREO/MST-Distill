import torch
from torch.utils.data import DataLoader
from torch import nn
from utils import get_data, get_dataset, get_Smodules, seed_all
import copy
import time
import os
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Dataset import SingleModalX
import sys


if __name__ == '__main__':
    '''
    Args Setting for CML.
    '''
    parser = ArgumentParser(description='CML-S')
    parser.add_argument('--database', type=str, default='AV-MNIST',
                        help="database name must be one of ['NYU-Depth-V2', 'RAVDESS', 'AV-MNIST', 'VGGSound-50k', 'CMMD-V2']")
    parser.add_argument('--Smodel', type=str, default='LeNet5',
                        help='Student model name')
    parser.add_argument('--mode', type=str, default='m1',
                        help='Data mode: m1 or m2')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for training')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers for DataLoader')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for training')
    parser.add_argument('--record', type=bool, default=True,
                        help='flag whether to record the learning log')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='cuda id')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs for training, default: 100-200')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='flag whether to save best model')
    parser.add_argument('--test_phase', type=bool, default=False,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--final_test', type=bool, default=True,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--commit', type=str, default='Stu-A',
                        help='Commit for logs')
    args = parser.parse_args()

    seed_all(args.seed)


    log_dir = f'./logs/stu'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = log_dir + f'/{args.database}_{args.lr}_{str(time.time()).split(".")[0]}_{args.Smodel}_{args.commit}'
    writer = SummaryWriter(log_dir, write_to_disk=args.record)

    data = get_data(args.database)
    data_train = get_dataset(args.database, data, 'train', args.seed)
    data_val = get_dataset(args.database, data, 'val', args.seed)
    data_test = get_dataset(args.database, data, 'test', args.seed)

    train_dataset = SingleModalX(data_train, args.database, mode=args.mode)
    valid_dataset = SingleModalX(data_val, args.database, mode=args.mode)
    test_dataset = SingleModalX(data_test, args.database, mode=args.mode)

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
    model_s, optimizer_s, scheduler_s, criterion_s, preprocessing, postprocessing, metric = get_Smodules(args)

    model_s = model_s.to(device)
    best_model_state = None
    best_val_loss = float('inf')
    print('================= Student Model Independent Training =================')
    for epoch in range(args.epochs):
        start_time1 = time.time()
        # train
        model_s.train()
        tra_LOSS_s = 0
        for i, (data, label) in tqdm(enumerate(train_loader), desc="Model Training ...",
                                            total=len(train_loader), dynamic_ncols=True,
                                            disable=False, file=sys.stdout):
            # print('Train Iter {}'.format(i))
            data, label = data.to(device), label.to(device)
            # outputs_t = model_t(data2)
            outputs_s = model_s(data) if preprocessing is None else preprocessing(model_s, data)
            loss_s = criterion_s(outputs_s, label) if postprocessing is None else postprocessing(outputs_s, label)
            optimizer_s.zero_grad()
            loss_s.backward()
            optimizer_s.step()
            tra_LOSS_s += loss_s.item()

        tra_LOSS_s_avg = tra_LOSS_s / (i + 1)
        print(f'Train =====> Epoch {epoch + 1}/{args.epochs}: loss_c = {tra_LOSS_s_avg:.4f}')
        writer.add_scalar('train/Loss', tra_LOSS_s_avg, epoch)

        # validation
        model_s.eval()
        metric.reset()
        L_s_val = 0
        acc_c = 0
        gt_list, pred_list = [], []
        with torch.no_grad():
            for i, (data, label) in tqdm(enumerate(valid_loader), desc="Model Validating ...",
                                                total=len(valid_loader), dynamic_ncols=True, disable=False, file=sys.stdout):
                # print('Val Iter {}'.format(i))
                data, label = data.to(device), label.to(device)
                outputs_s = model_s(data) if preprocessing is None else preprocessing(model_s, data)
                loss_s = criterion_s(outputs_s, label) if postprocessing is None else postprocessing(outputs_s, label)
                L_s_val += loss_s.item()
                metric.update(outputs_s, label)
            L_s_val = L_s_val / (i + 1)
        # res = metric.compute()
        # print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss_c = {L_s_val:.4f}, OA_c = {res['Accuracy']:.2f}%")
        # writer.add_scalar('valid/Loss', L_s_val, epoch)
        # writer.add_scalar('valid/Acc', res['Accuracy'], epoch)

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
            L_s = 0
            acc_c = 0
            gt_list, pred_list = [], []
            with torch.no_grad():
                for i, (data, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                                    total=len(test_loader), dynamic_ncols=True, disable=False, file=sys.stdout):
                    # print('Val Iter {}'.format(i))
                    data, label = data.to(device), label.to(device)
                    outputs_s = model_s(data) if preprocessing is None else preprocessing(model_s, data)
                    loss_s = criterion_s(outputs_s, label) if postprocessing is None else postprocessing(outputs_s, label)
                    L_s = L_s + loss_s.item()
                    metric.update(outputs_s, label)
                L_s = L_s / (i + 1)
            res = metric.compute()
            print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss_c = {L_s:.4f}, OA_c = {res['Accuracy']}")
            writer.add_scalar('test/Loss', L_s, epoch)
            writer.add_scalar('test/Acc', float(res['Accuracy']), epoch)

            # For NYU-Depth-V2
            # print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_s:.4f}")
            # writer.add_scalar('test/Loss', L_s, epoch)

            if (epoch + 1) % 10 == 0:
                print('\n===============Metrics==================')
                for e in res.keys():
                    print(e)
                    print(res[e])
                    print('----------------------------')
                print('=======================================\n')

            # scheduler_t.step()

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
        metric.reset()
        gt_list, pred_list = [], []
        with torch.no_grad():
            for i, (data, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                         total=len(test_loader), dynamic_ncols=True, disable=True, file=sys.stdout):
                data, label = data.to(device), label.to(device)
                outputs_s = model_s(data) if preprocessing is None else preprocessing(model_s, data)
                metric.update(outputs_s, label)
        res = metric.compute()

        print('\n===============Metrics==================')
        for e in res.keys():
            print(e)
            print(res[e])
            print('----------------------------')
        print('=======================================\n')

    if args.save_model:
        if not os.path.exists('./checkpoints/stu/wo_kd'):
            os.makedirs('./checkpoints/stu/wo_kd')
        torch.save(best_model_state, f'./checkpoints/stu/wo_kd/{args.database}_{args.Smodel}_seed{args.seed}_{args.mode}_ep{best_epoch+1}-{args.epochs}.pth')