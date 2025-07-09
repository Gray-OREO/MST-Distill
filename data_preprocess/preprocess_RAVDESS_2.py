import numpy as np
import os
import librosa
from collections import Counter


def video_loader(video_dir_path):
    video = np.load(video_dir_path)  # [N, H, W, C]
    video_data = video.transpose(0, 3, 1, 2)
    return video_data


def load_audio(audiofile, sr=22050):
    audios, sr = librosa.core.load(audiofile, sr=sr)
    mfcc = librosa.feature.mfcc(y=audios, sr=sr, n_mfcc=15)
    return mfcc


if __name__ == '__main__':
    # data1 = video_loader('E:/Gray/Database/RAVDESS-Speech/Actor_01/01-01-01-01-01-01-01_facecroppad.npy')
    # data2 = load_audio('E:/Gray/Database/RAVDESS-Speech/Actor_01/03-01-01-01-01-01-01_croppad.wav')

    video_list = []
    tar_root = 'E:/Gray/Database/RAVDESS_preprocessed_npy/'
    os.makedirs(tar_root, exist_ok=True)
    for root, dirs, files in os.walk('E:/Gray/Database/RAVDESS-preprocessed/'):
        for file in files:
            if file.endswith('.npy'):
                video_list.append(root+'/'+file)

    video_data, audio_data, label_data = [], [], []
    for i in range(len(video_list)):
        """
        root = 'E:/Gray/Database/RAVDESS-Speech/Actor_01/'
        sample_name = '01-01-01-01-01-01-01'
        'facecroppad.npy'
        """
        root = video_list[i].split('_')[0]+'_'+video_list[i].split('_')[1].split('/')[0]+'/'
        sample_name = video_list[i].split('_')[1].split('/')[1]
        no_mode_name = sample_name[2:]
        label = int(sample_name.split('-')[2]) - 1
        video_dir = root+'01'+no_mode_name+'_facecroppad.npy'
        audio_dir = root+'03'+no_mode_name+'_croppad.wav'
        video_data.append(video_loader(video_dir))
        audio_data.append(load_audio(audio_dir))
        label_data.append(label)
    np.save(f'{tar_root}video_data.npy', np.array(video_data))
    np.save(f'{tar_root}audio_data.npy', np.array(audio_data))
    np.save(f'{tar_root}label_data.npy', np.array(label_data))
    # # 使用Counter统计元素出现次数
    # counter = Counter(label_data)
    #
    # # 输出结果
    # print(counter)
