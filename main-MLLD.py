import torch
from torch.utils.data import DataLoader
from utils import get_data, get_dataset, get_Tmodules, get_Smodules, seed_all
import copy
import time
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Dataset import MultiModalX
from KD_methods.MLLD import MultiLevelLogitDistillation
import sys
import os


if __name__ == '__main__':
    '''
    Args Setting for CML.
    '''
    parser = ArgumentParser(description='CML-MLLD')
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
    parser.add_argument('--batch_size', type=int, default=512,
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
    parser.add_argument('--final_test', type=bool, default=False,
                        help='flag whether to conduct the test phase')
    parser.add_argument('--commit', type=str, default='MLLD',
                        help='Commit for logs')
    args = parser.parse_args()

    seed_all(args.seed)

    # 保存log
    log_dir = f'./logs/mlld'
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
    model_s, optimizer_s, scheduler_s, criterion_s, preprocessing_s, postprocessing_s, metric = get_Smodules(args)
    mlld_loss = MultiLevelLogitDistillation()

    model_t = model_t.to(device)
    model_s = model_s.to(device)
    model_t.load_state_dict(torch.load(f'checkpoints/tea/{args.ckpt_name}', weights_only=True, map_location=f'cuda:{args.cuda_id}'))
    model_t.eval()

    best_model_state = None
    best_val_loss = float('inf')
    # Cloud-only Learning
    print('================= CML-KD =================')
    for epoch in range(args.epochs):
        start_time1 = time.time()
        # train
        model_s.train()

        tra_LOSS_s, tra_LOSS_task, tra_LOSS_kl = 0, 0 ,0
        for i, (data, data2, label) in tqdm(enumerate(train_loader), desc="Model Training ...",
                                            total=len(train_loader), dynamic_ncols=True,
                                            disable=False, file=sys.stdout):
            # print('Train Iter {}'.format(i))
            data, data2, label = data.to(device), data2.to(device), label.to(device)
            data_s = data if args.mode == 'm1' else data2
            outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
            outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)
            loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
            loss_kl = mlld_loss(outputs_s, outputs_t)
            loss = (1.0 - args.alpha) * loss_s + args.alpha * loss_kl
            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()
            tra_LOSS_s += loss.item()
            tra_LOSS_task += loss_s.item()
            tra_LOSS_kl += loss_kl.item()

        tra_LOSS_s_avg = tra_LOSS_s / (i + 1)
        tra_LOSS_task_avg = tra_LOSS_task / (i + 1)
        tra_LOSS_kl_avg = tra_LOSS_kl / (i + 1)
        print(f'Train =====> Epoch {epoch + 1}/{args.epochs}: loss = {tra_LOSS_s_avg:.4f} | '
              f'loss_task = {tra_LOSS_task_avg:.4f} | loss_mlld = {tra_LOSS_kl_avg:.4f}')
        writer.add_scalar('train/Loss', tra_LOSS_s_avg, epoch)

        # validation
        model_s.eval()
        metric.reset()
        L_s_val = 0
        acc_c = 0
        Loss_s, Loss_kl = 0, 0
        gt_list, pred_list = [], []
        with torch.no_grad():
            for i, (data, data2, label) in tqdm(enumerate(valid_loader), desc="Model Validating ...",
                                                total=len(valid_loader), dynamic_ncols=True, disable=False,
                                                file=sys.stdout):
                # print('Val Iter {}'.format(i))
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                data_s = data if args.mode == 'm1' else data2
                outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)
                loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
                loss_kl = mlld_loss(outputs_s, outputs_t)
                loss = loss_s
                L_s_val += loss.item()
                Loss_kl += loss_kl.item()
                Loss_s += loss_s.item()
                metric.update(outputs_s, label)
            L_s_val = L_s_val / (i + 1)
            Loss_kl_avg = Loss_kl / (i + 1)
            Loss_s_avg = Loss_s / (i + 1)
        # res = metric.compute()
        # print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: OA_s = {float(res['Accuracy'])} | loss = {L_s_val:.4f} | "
        #       f"Loss_s = {Loss_s_avg:.4f} | Loss_kl = {Loss_kl_avg:.4f}")
        # writer.add_scalar('valid/Loss', L_s_val, epoch)
        # writer.add_scalar('valid/Acc', float(res['Accuracy']), epoch)

        # For NYU-Depth-V2
        print(f"Valid =====> Epoch {epoch + 1}/{args.epochs}: loss = {L_s_val:.4f} | loss_task = {Loss_s_avg:.4f} | loss_mlld = {Loss_kl_avg:.4f}")
        writer.add_scalar('valid/Loss', L_s_val, epoch)

        # 保存验证集上表现最好的模型
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
            Loss_s, Loss_kl = 0, 0
            gt_list, pred_list = [], []
            with torch.no_grad():
                for i, (data, data2, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                                    total=len(test_loader), dynamic_ncols=True, disable=False,
                                                    file=sys.stdout):
                    data, data2, label = data.to(device), data2.to(device), label.to(device)
                    data_s = data if args.mode == 'm1' else data2
                    outputs_t = model_t(data, data2) if preprocessing_t is None else preprocessing_t(model_t, data, data2)
                    outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)
                    loss_s = criterion_s(outputs_s, label) if postprocessing_s is None else postprocessing_s(outputs_s, label)
                    loss_kl = mlld_loss(outputs_s, outputs_t)
                    loss = (1.0 - args.alpha) * loss_s + args.alpha * loss_kl
                    L_t = L_t + loss.item()
                    Loss_kl += loss_kl.item()
                    Loss_s += loss_s.item()
                    metric.update(outputs_s, label)
                L_t = L_t / (i + 1)
                Loss_kl_avg = Loss_kl / (i + 1)
                Loss_s_avg = Loss_s / (i + 1)
            # res = metric.compute()
            # print(f"Test  =====> Epoch {epoch + 1}/{args.epochs}: OA_s = {float(res['Accuracy'])} | loss = {L_t:.4f} | "
            #       f"Loss_s = {Loss_s_avg:.4f} | Loss_kl = {Loss_kl_avg:.4f}")
            # writer.add_scalar('test/Loss', L_t, epoch)
            # writer.add_scalar('test/Acc', float(res['Accuracy']), epoch)

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

            # scheduler_t.step()  # 学习率衰减(当训练SAFN模型时，需加入监听指标作为参数)
        args.alpha *= 0.5 if (epoch+1) % 30 == 0 else 1.0
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
            for i, (data, data2, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                         total=len(test_loader), dynamic_ncols=True, disable=True, file=sys.stdout):
                data, data2, label = data.to(device), data2.to(device), label.to(device)
                data_s = data if args.mode == 'm1' else data2
                outputs_s = model_s(data_s) if preprocessing_s is None else preprocessing_s(model_s, data_s)
                metric.update(outputs_s, label)
        res = metric.compute()

        print('\n===============Metrics==================')
        for e in res.keys():
            print(e)
            print(res[e])
            print('----------------------------')
        print('=======================================\n')

    if args.save_model:
        if not os.path.exists('./checkpoints/stu/mlld'):
            os.makedirs('./checkpoints/stu/mlld')
        names = args.ckpt_name.split('_')
        Tmodel_mode = names[3]
        torch.save(best_model_state,
                   f'./checkpoints/stu/mlld/{args.database}_{args.Tmodel}_{Tmodel_mode}--{args.Smodel}_seed{args.seed}_{args.mode}_ep{best_epoch + 1}-{args.epochs}.pth')
