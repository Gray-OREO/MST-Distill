import json
import numpy as np
import os
from tqdm import tqdm

'''
处理segment标签的代码
'''

output_path = 'E:/Gray/Database/VGGSound-AVEL50k/seg_label/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 读取JSON文件内容
with open('E:/Gray/Database/VGGSound-AVEL50k/vggsound-avel50k_labels.json', 'r') as file:
    data = json.load(file)

with open('D:/Gray/Database/VGGS50K/VGGS50k_metadata.txt', 'r') as txt_file:
    txt_info = txt_file.read().splitlines()

vid2vinfo = {}
for item in txt_info:
    vid = item[1:12]
    sample_name = item.split('&')[0]
    vid2vinfo.update({vid: sample_name})

for item in tqdm(data, total=len(data), desc='Processing...'):
    for video_id, info in item.items():
        # 获取label数据并转换为ndarray
        label = np.array(info['label']).reshape(1, 10)
        # 将ndarray保存为.npy文件
        try:
            sample_name = vid2vinfo[video_id]
            np.save(f'{output_path}{sample_name}_sLabel.npy', label)
        except:
            pass
