import time
import torch
from utils import get_data, get_dataset, data_preprocessing
from torch.utils.data import DataLoader
import numpy as np


class MultiModalX(torch.utils.data.Dataset):
    """ Generic class for a MultiModal Data """

    def __init__(self, data, data_name, mode=None, for_contrast=False, k=4096):
        """
        Args:
            data_name: [str] database name)
        """
        super(MultiModalX, self).__init__()
        self.dataset = data_name
        self.data = data[0]
        self.data1 = data[1]
        self.labels = data[2]
        self.mode = mode
        self.for_contrast = for_contrast
        self.k = k

        if self.for_contrast:
            n = len(self.data)
            num_cls = get_ClsNum(data_name)
            self.cls_positive = [[] for _ in range(num_cls)]
            for i in range(n):
                if self.dataset == 'VGGSound-50k':
                    self.cls_positive[self.labels[i][0, -1].long()].append(i)
                else:
                    self.cls_positive[self.labels[i]].append(i)
            self.cls_negative = [[] for _ in range(num_cls)]
            for i in range(num_cls):
                for j in range(num_cls):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])
            self.cls_positive = np.asarray([np.asarray(self.cls_positive[i]) for i in range(num_cls)], dtype=object)
            self.cls_negative = np.asarray([np.asarray(self.cls_negative[i]) for i in range(num_cls)], dtype=object)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the total data from memory
        m1_data, m2_data, gt = self.data[idx], self.data1[idx], self.labels[idx]
        # ================================================================================
        m1_data, m2_data = data_preprocessing(m1_data, m2_data, self.mode)
        if not self.for_contrast:
            return m1_data, m2_data, gt
        else:
            pos_idx = idx
            cls_neg_idx = gt[0, -1].long() if self.dataset == 'VGGSound-50k' else gt.long()
            replace = True if self.k > len(self.cls_negative[cls_neg_idx]) else False
            neg_idx = np.random.choice(self.cls_negative[cls_neg_idx], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return m1_data, m2_data, gt, idx, sample_idx


class SingleModalX(torch.utils.data.Dataset):
    """ Generic class for a MultiModal Data """

    def __init__(self, data, data_name, mode=None):
        """
        Args:
            dataname: [str] database name)
        """
        super(SingleModalX, self).__init__()
        self.dataset = data_name
        self.labels = data[2]
        self.mode = mode
        if mode in ['m1', 'm2']:
            # print(f'Data loading mode: {mode}')
            self.data = data[0] if mode == 'm1' else data[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the sample from memory for VGGSound-50k====================================
        # if self.dataset == 'VGGSound-50k_old':
        #     fea = np.load(self.data[idx]).astype(np.float32) if self.mode == 'm1' else np.load(self.data[idx])
        #     label = self.labels[idx].astype(np.float32)
        #     data, gt = torch.from_numpy(fea), torch.from_numpy(label)
        # else:
        #     data, gt = self.data[idx], self.labels[idx]
        # ================================================================================
        # Load the total data from memory
        data, gt = self.data[idx], self.labels[idx]
        return data, gt


def get_ClsNum(data_name):
    if data_name == 'AV-MNIST':
        return 10
    elif data_name == 'NYU-Depth-V2':
        return 41
    elif data_name == 'RAVDESS':
        return 8
    elif data_name == 'VGGSound-50k':
        return 141
    elif data_name == 'CMMD-V2':
        return 8
    else:
        raise ValueError(f'Invalid data name: {data_name}')


if __name__ == '__main__':
    dataname = 'NYU-Depth-V2'
    data = get_data(dataname)
    data_train = get_dataset(dataname, data, 'train', 0)
    train_dataset = MultiModalX(data_train, dataname, mode='none')
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        pin_memory=True,
        # num_workers=16,
        shuffle=True
    )

    t0 = time.time()
    for i, datas in enumerate(train_loader):
        t1 = time.time()
        t = t1 - t0
        t0 = t1
        print(f'Time cost:{t:.2f}')
        # print(datas[0].shape, datas[1].shape, datas[2].shape)
