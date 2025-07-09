from utils import get_data, seed_all
import numpy as np
import pandas as pd
import os


def generate_random_indices(data_length, num_groups=20, output_file="random_indices.csv"):
    # 创建一个空的DataFrame来存放随机索引
    random_indices = pd.DataFrame()

    # 生成 num_groups 组随机索引
    for i in range(num_groups):
        indices = np.random.permutation(data_length)  # 生成一个乱序的索引数组
        random_indices[f'group_{i}'] = indices  # 将该数组作为新列添加到DataFrame

    # 保存为csv文件
    random_indices.to_csv(output_file, index=False)
    print(f"{output_file} generated successfully!")


if __name__ == "__main__":
    dataset_name = 'CMMD-V2'
    save_path = 'metadata/'
    os.makedirs(save_path, exist_ok=True)
    seed_all(19980427)  # 设置随机种子
    data1, data2, labels = get_data(dataset_name)
    num_data = len(data1)
    generate_random_indices(num_data, num_groups=20, output_file=f"{save_path}{dataset_name}_indices.csv")