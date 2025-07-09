from utils import _obtain_avel_label
import platform
from tqdm import tqdm
import numpy as np
import torch
import os

def get_vggsound(rootpath):
    root = rootpath + 'VGGS50K/' if platform.system() == 'Linux' else 'D:/Gray/Database/VGGS50K/'
    vf_list, af_list, labels = [], [], []
    with open(root + 'VGGS50k_metadata.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 读取所有行并保存为列表
        data = [line.strip() for line in lines]  # 去掉每行末尾的换行符
        # data = data[:1000]  # For debug(failed for contrastive learning)
        for line in tqdm(data, total=len(data), desc='Data loading...'):
            sample_name = line.split('&')[0]
            label = int(line.split('&')[1])
            vf = root + 'VGGS50K_features/visual_features/' + f'{sample_name}_vFeature.npy'
            af = root + 'VGGS50K_features/audio_features/' + f'{sample_name}_aFeature.npy'
            avc_label = np.load(root + 'seg_labels/' + f'{sample_name}_sLabel.npy')
            slabel = _obtain_avel_label(avc_label, label)
            vf_list.append(torch.from_numpy(np.load(vf).astype(np.float32)))
            af_list.append(torch.from_numpy(np.load(af)))
            labels.append(torch.from_numpy(slabel.astype(np.float32)))
    return vf_list, af_list, labels


def get_max_min(data_list):
    all_values = np.concatenate([tensor.flatten() for tensor in data_list])
    global_min = all_values.min()
    global_max = all_values.max()
    return global_min, global_max


if __name__ == '__main__':
    root = 'D:/Gray/Database/' if platform.system() == 'Windows' else '/home/gray/Database/'
    save_path_v = root + 'VGGS50K/VGGS50K_features_normed/visual_features/'
    save_path_a = root + 'VGGS50K/VGGS50K_features_normed/audio_features/'

    os.makedirs(save_path_v, exist_ok=True)
    os.makedirs(save_path_a, exist_ok=True)

    vf_list, af_list, _ = get_vggsound(root)
    v_min, v_max = get_max_min(vf_list)
    a_min, a_max = get_max_min(af_list)

    with open(root + 'VGGS50K/VGGS50k_metadata.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 读取所有行并保存为列表
        data = [line.strip() for line in lines]  # 去掉每行末尾的换行符
        for line in tqdm(data, total=len(data), desc='Data norm & save...'):
            sample_name = line.split('&')[0]
            label = int(line.split('&')[1])
            vf = root + 'VGGS50K/VGGS50K_features/visual_features/' + f'{sample_name}_vFeature.npy'
            af = root + 'VGGS50K/VGGS50K_features/audio_features/' + f'{sample_name}_aFeature.npy'
            vf = np.load(vf)
            af = np.load(af)
            vf = (vf - v_min) / (v_max - v_min)
            af = (af - a_min) / (a_max - a_min)
            np.save(save_path_v + f'{sample_name}_vFeature.npy', vf)
            np.save(save_path_a + f'{sample_name}_aFeature.npy', af)

