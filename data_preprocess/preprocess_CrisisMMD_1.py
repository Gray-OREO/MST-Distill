import os
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# 定义映射函数
def map_category(value):
    """
    - 输入整数 (0-7)，返回类别名称
    - 输入类别名称，返回整数 (0-7)
    - 处理无效输入
    """
    if isinstance(value, int):
        return index_to_category.get(value, "Unknown Category")  # 防止索引越界
    elif isinstance(value, str):
        return category_to_index.get(value, -1)  # 未知类别返回 -1
    else:
        raise ValueError("Input must be an integer (0-7) or a valid category name.")


def text_feature_extractor(text):
    with torch.no_grad():
        features = bertweet(text)  # Models outputs are now tuples
    return features.pooler_output.squeeze(0).detach().cpu().numpy()


def im_feature_extractor(image_path, device):
    image = Image.open(image_path).convert("RGB")  # 确保图像是 RGB 模式
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet50(image.to(device))  # Models outputs are now tuples
    return features.squeeze(0).squeeze(-1).squeeze(-1).detach().cpu().numpy()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ====================================== TEXT =====================================
categories = [
    'not_humanitarian',  # 0 6179
    'affected_individuals',  # 1 521
    'infrastructure_and_utility_damage',  # 2 2150
    'injured_or_dead_people',  # 3 301
    'missing_or_found_people',  # 4 32
    'rescue_volunteering_or_donation_effort',  # 5 2594
    'vehicle_damage',  # 6 160
    'other_relevant_information'  # 7 4121
]

# 创建索引到类别的映射
index_to_category = {i: cat for i, cat in enumerate(categories)}
category_to_index = {cat: i for i, cat in enumerate(categories)}
# ================================= IMG ===========================================
resnet50 = models.resnet50(pretrained=True).to(device)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])  # 移除最后的分类层
resnet50.eval()  # 设置为评估模式

# 2. 定义图像预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整到 ResNet 输入大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])


data_file = 'E:/Gray/Database/CrisisMMD_v2.0/all_data.csv'
root = 'E:/Gray/Database/CrisisMMD_v2.0'
textpath = 'E:/Gray/Database/CrisisMMD_v2.0/crisismmd_datasplit_all'

data = pd.read_csv(data_file, sep='\t', encoding="utf-8")
category = set(data['label'])
# print(data)

bertweet = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# INPUT TWEET IS ALREADY NORMALIZED!
line = "DHEC confirms HTTPURL via @USER :crying_face:"

text_inputs = []
text_fs = []
im_paths = []
im_fs = []
labels = []

for im_path, text, label in tqdm(zip(data['image'], data['tweet_text'], data['label']), total=len(data), desc='Data Processing'):
    input_ids = torch.tensor([tokenizer.encode(text, max_length=128)], device=device)
    text_f = text_feature_extractor(input_ids)
    text_fs.append(text_f)

    im_path = os.path.join(root, im_path)
    im_f = im_feature_extractor(im_path, device)
    im_fs.append(im_f)

    labels.append(map_category(label))

print(len(text_fs))
text_fs = np.array(text_fs)
im_fs = np.array(im_fs)
labels = np.array(labels)
res = (im_fs, text_fs, labels)
torch.save(res, f"{root}/CMMD_data.pth")
print('All Done!')
exit()