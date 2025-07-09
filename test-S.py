import torch
from torch.utils.data import DataLoader
from utils import get_data, get_dataset, get_Smodules, seed_all
from argparse import ArgumentParser
from tqdm import tqdm
from Dataset import SingleModalX
import sys


if __name__ == '__main__':
    '''
    Args Setting for CML.
    '''
    parser = ArgumentParser(description='CML-SO')
    parser.add_argument('--database', type=str, default='CMMD-V2',
                        help="database name must be one of "
                             "['NYU-Depth-V2', 'RAVDESS', 'AV-MNIST', 'VGGSound-50k', 'MM-IMDb']")
    parser.add_argument('--Smodel', type=str, default='MLP-Vb',
                        help='Student model name')

    parser.add_argument('--KD_method', type=str, default='KD',
                        help='The name of the dictionary to be loaded in ./checkpoints/stu/...')
    parser.add_argument('--ckpt_name', type=str,
                        default='Your checkpoint name',
                        help='The name of the weight to be loaded in ./checkpoints/stu/{KD method}/...')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--mode', type=str, default='m1',
                        help='Data mode: m1 or m2')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for training')
    parser.add_argument('--record', type=bool, default=True,
                        help='flag whether to record the learning log')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='cuda id')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs for training, invalid in this file but for get_module runing.')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='flag whether to save best model')
    parser.add_argument('--commit', type=str, default='MM-T',
                        help='Commit for logs')
    args = parser.parse_args()

    seed_all(args.seed)

    data = get_data(args.database)
    data_test = get_dataset(args.database, data, 'test', args.seed)
    test_dataset = SingleModalX(data_test, args.database, mode=args.mode)

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
    model_s.load_state_dict(torch.load(f'checkpoints/stu/{args.KD_method}/{args.ckpt_name}', map_location=device, weights_only=True))
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

