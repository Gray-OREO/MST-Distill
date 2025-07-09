import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from utils import *

import numpy as np
import torch.optim as optim
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
    parser = ArgumentParser(description='CML-TO')
    parser.add_argument('--database', type=str, default='AV-MNIST',
                        help="database name must be one of ['NYU-Depth-V2', 'RAVDESS', 'AV-MNIST', 'VGGSound-50k', 'MM-IMDb']")
    parser.add_argument('--Tmodel', type=str, default='CNN-I',
                        help='Teacher model name')

    parser.add_argument('--ckpt_name', type=str,
                        default='AV-MNIST_CNN-I_seed0_ORG_ep97-100.pth',
                        help='The name of the weight to be loaded in ./checkpoints/tea')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--mode', type=str, default='ORG',
                        help='Data mode: ORG or m1-MSK-0.1 or m2-GN-0.01')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epoch for training, invalid in this file but for get_module runing.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for training')
    parser.add_argument('--record', type=bool, default=True,
                        help='flag whether to record the learning log')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='cuda id')
    parser.add_argument('--freeze_bn', type=bool, default=True,
                        help='flag whether to freeze BN layers in the model')
    args = parser.parse_args()

    seed_all(args.seed)

    data = get_data(args.database)
    data_test = get_dataset(args.database, data, 'test', args.seed)
    test_dataset = MultiModalX(data_test, args.database, mode=args.mode)

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
    model_t.load_state_dict(torch.load(f'checkpoints/tea/{args.ckpt_name}', map_location=device, weights_only=True))
    # test
    model_t.eval()
    metric.reset()
    gt_list, pred_list = [], []
    with torch.no_grad():
        for i, (data, data2, label) in tqdm(enumerate(test_loader), desc="Model Testing ...",
                                            total=len(test_loader), dynamic_ncols=True, disable=True, file=sys.stdout):
            data, data2, label = data.to(device), data2.to(device), label.to(device)
            outputs_t = model_t(data, data2) if preprocessing is None else preprocessing(model_t, data, data2)
            metric.update(outputs_t, label)
    res = metric.compute()

    print('\n===============Metrics==================')
    for e in res.keys():
        print(e)
        print(res[e])
        print('----------------------------')
    print('=======================================\n')

