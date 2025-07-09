import torch
from torch.utils.data import DataLoader
import os
from utils import seed_all, get_data, get_dataset, get_Tmodules, optimizers_zero_grad, optimizers_step
import copy
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
    parser = ArgumentParser(description='CML-T')
    parser.add_argument('--database', type=str, default='AV-MNIST',
                        help="database name must be one of "
                             "['NYU-Depth-V2', 'RAVDESS', 'AV-MNIST', 'VGGSound-50k', 'CMMD-V2']")
    parser.add_argument('--Tmodel', type=str, default='CNN-I',
                        help='Teacher model name')
    parser.add_argument('--mode', type=str, default='ORG',
                        help='Data mode: ORG or m1 or m2')
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
                        help='epochs for training, default: 100')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='flag whether to save best model')
    parser.add_argument('--test_phase', type=bool, default=False,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--final_test', type=bool, default=True,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--commit', type=str, default='MM-T',
                        help='Commit for logs')
    args = parser.parse_args()

    seed_all(args.seed)

    log_dir = f'./logs/tea'
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
    test_dataset = MultiModalX(data_test, args.database, mode='ORG')

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
    model_t, optimizers, scheduler_t, criterion, preprocessing, postprocessing, metric = get_Tmodules(args, device)

    model_t = model_t.to(device)
    best_model_state = None
    best_val_loss = float('inf')
    print('================= Teacher Model Independent Training =================')
    for epoch in range(args.epochs):
        start_time1 = time.time()
        # train
        model_t.train()

        tra_LOSS_t = 0
        for i, (data, data2, label) in tqdm(enumerate(train_loader), desc="Model Training ...",
                                            total=len(train_loader), dynamic_ncols=True,
                                            disable=False, file=sys.stdout):
            # print('Train Iter {}'.format(i))
            data, data2, label = data.to(device), data2.to(device), label.to(device)
            # outputs_t = model_t(data2)
            outputs_t = model_t(data, data2) if preprocessing is None else preprocessing(model_t, data, data2)
            loss_t = criterion(outputs_t, label) if postprocessing is None else postprocessing(outputs_t, label)
            optimizers_zero_grad(optimizers)
            loss_t.backward()
            optimizers_step(optimizers, epoch)
            tra_LOSS_t += loss_t.item()

        tra_LOSS_c_avg = tra_LOSS_t / (i + 1)
        print(f'Train =====> Epoch {epoch + 1}/{args.epochs}: loss = {tra_LOSS_c_avg:.4f}')
        writer.add_scalar('train/Loss', tra_LOSS_c_avg, epoch)

        # validation
        model_t.eval()
        metric.reset()
        L_t_val = 0
        acc_c = 0
        gt_list, pred_list = [], []
        with torch.no_grad():
            for i, (data, data2, label) in tqdm(enumerate(valid_loader), desc="Model Validating ...",
                                                total=len(valid_loader), dynamic_ncols=True, disable=False, file=sys.stdout):
                # print('Val Iter {}'.format(i))
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                # outputs_t = model_t(data2)
                outputs_t = model_t(data, data2) if preprocessing is None else preprocessing(model_t, data, data2)

                loss_t = criterion(outputs_t, label) if postprocessing is None else postprocessing(outputs_t, label)
                L_t_val += loss_t.item()
                metric.update(outputs_t, label)
            L_t_val = L_t_val / (i + 1)
        # res = metric.compute()
        # print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss_t = {L_t_val:.4f}, OA_t = {res['Accuracy']:.2f}%")
        # writer.add_scalar('valid/Loss', L_t_val, epoch)
        # writer.add_scalar('valid/Acc', res['Accuracy'], epoch)

        # For NYU-Depth-V2
        print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_t_val:.4f}")
        writer.add_scalar('valid/Loss', L_t_val, epoch)

        if L_t_val < best_val_loss:
            best_val_loss = L_t_val
            best_model_state = copy.deepcopy(model_t.state_dict())
            best_epoch = epoch

        # test
        if args.test_phase:
            model_t.eval()
            metric.reset()
            L_t = 0
            acc_c = 0
            gt_list, pred_list = [], []
            with torch.no_grad():
                for i, (data, data2, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                                    total=len(test_loader), dynamic_ncols=True, disable=False, file=sys.stdout):
                    # print('Val Iter {}'.format(i))
                    data, data2, label = data.to(device), data2.to(device), label.to(device)
                    # outputs_t = model_t(data2)
                    outputs_t = model_t(data, data2) if preprocessing is None else preprocessing(model_t, data, data2)

                    loss_t = criterion(outputs_t, label) if postprocessing is None else postprocessing(outputs_t, label)
                    L_t = L_t + loss_t.item()
                    metric.update(outputs_t, label)
                L_t = L_t / (i + 1)
            res = metric.compute()
            print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss_t = {L_t:.4f}, OA_t = {res['Accuracy']:.4f}")
            writer.add_scalar('test/Loss', L_t, epoch)
            writer.add_scalar('test/Acc', float(res['Accuracy']), epoch)

            # # For NYU-Depth-V2
            # print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_t:.4f}")
            # writer.add_scalar('test/Loss', L_t, epoch)

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
        model_t.load_state_dict(best_model_state)
        print('================= Final Test for This Model =================')
        # test
        model_t.eval()
        metric.reset()
        gt_list, pred_list = [], []
        with torch.no_grad():
            for i, (data, data2, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                         total=len(test_loader), dynamic_ncols=True, disable=True, file=sys.stdout):
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                # outputs_t = model_t(data2)
                outputs_t = model_t(data, data2) if preprocessing is None else preprocessing(model_t, data, data2)
                metric.update(outputs_t, label)
        res = metric.compute()

        print('\n===============Metrics==================')
        for e in res.keys():
            print(e)
            print(res[e])
            print('----------------------------')
        print('=======================================\n')

    if args.save_model:
        if not os.path.exists('./checkpoints/tea'):
            os.makedirs('./checkpoints/tea')
        torch.save(best_model_state, f'./checkpoints/tea/{args.database}_{args.Tmodel}_seed{args.seed}_{args.mode}_ep{best_epoch+1}-{args.epochs}.pth')