import pandas as pd
import os
import tqdm
import subprocess
import logging
import numpy as np


def read_txt_to_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 读取所有行并保存为列表
        lines = [line.strip() for line in lines]  # 去掉每行末尾的换行符
    return lines


if __name__ == '__main__':
    # data_a = np.load('E:/Gray/Project/feature_extractor/VGGS50K_features/audio_features/v6bPsB1h3wvE_110_120_out_aFeature.npy')
    # data_v = np.load('E:/Gray/Project/feature_extractor/VGGS50K_features/visual_features/v6bPsB1h3wvE_110_120_out_vFeature.npy')
    # print(data_a.shape)
    # print(data_v.shape)

    with open('D:/Gray/Database/VGGS50K/VGGS50K_videos.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 读取所有行并保存为列表
        data = [line.strip() for line in lines]  # 去掉每行末尾的换行符

    feature_path = 'E:/Gray/Project/feature_extractor/VGGS50K_features/'
    v_features, a_features = [], []
    for root, dirs, files in os.walk(feature_path+'visual_features'):
        for file in files:
            v_features.append(file[1:12])
    for root, dirs, files in os.walk(feature_path+'audio_features'):
        for file in files:
            a_features.append(file[1:12])

    for line in data:
        sample_id = line.split('/')[6].split('.')[0][1:12]
        file_name = line.split('/')[6].split('.')[0]
        label = line.split('&')[-1]
        if sample_id in v_features and sample_id in a_features:
            with open('D:/Gray/Database/VGGS50K/VGGS50k_metadata.txt', 'a', encoding='utf-8') as f:
                f.write(f'{file_name}&{label}\n')




