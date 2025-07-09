import os
import pandas as pd
from emoji import demojize
import ftfy
import re
from nltk.tokenize import TweetTokenizer


tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):  # 用户名替换
        return "@USER"
    elif lowercased_token.startswith("https://t.co/"):  # 删除特定 URL
        return ""
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):  # 其他 URL 替换
        return "HTTPURL"
    elif len(token) == 1:  # 单字符处理（可能是表情符号）
        return demojize(token)
    else:  # 特殊字符替换
        token = token.replace("\\'", "'")  # 处理反斜杠引号
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tweet = tweet.replace("\\'", "'")  # 处理 Windows 下的转义问题
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])
    # 处理 \' 为 '
    normTweet = re.sub(r"\\'", "'", normTweet)
    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace(r"'re ", " 're ")
        .replace(r"'s ", " 's ")
        .replace(r"'ll ", " 'll ")
        .replace(r"'d ", " 'd ")
        .replace(r"'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


def safe_unicode_decode(text):
    """ 仅当字符串包含 Unicode 转义字符时才进行 decode 处理 """
    if isinstance(text, str):
        if '\\u' in text or '\\U' in text:  # 仅处理包含 \u 或 \U 的文本
            try:
                return text.encode('utf-8').decode('unicode-escape')
            except UnicodeDecodeError:
                return text  # 解析失败，返回原始文本
        else:
            return text  # 直接返回
    return text  # 非字符串类型，保持不变


def fix_unicode_escapes_1(text):
    # 先将字符串转换回 Unicode escape 格式
    unicode_escaped_text = text.encode('unicode-escape').decode('utf-8')
    # 替换 \uXXXX 形式为 \U000XXXXX 以确保兼容性
    return re.sub(r'\\u([0-9a-fA-F]{4,6})', lambda m: '\\U' + m.group(1).zfill(8), unicode_escaped_text)


root_ = 'E:/Gray/Database/CrisisMMD_v2.0'
impath = 'E:/Gray/Database/CrisisMMD_v2.0/data_image'
textpath = 'E:/Gray/Database/CrisisMMD_v2.0/crisismmd_datasplit_all'

ims = []
for root, dirs, files in os.walk(impath):
    for file in files:
        if file.endswith('.jpg'):
            ims.append(file)

ano_files = []
for root, dirs, files in os.walk(textpath):
    for file in files:
        if file.startswith('task_humanitarian'):
            ano_files.append(file)

# print(len(ano_files))

datas = []
for file in ano_files:
    data = pd.read_csv(os.path.join(textpath, file), sep='\t', encoding="utf-8")
    datas.append(data)

ano_data = pd.concat(datas, ignore_index=True)
ano_data["tweet_text"] = ano_data["tweet_text"].apply(lambda x: ftfy.fix_text(x))
ano_data["tweet_text"] = ano_data["tweet_text"].apply(lambda x: fix_unicode_escapes_1(x))
ano_data["tweet_text"] = ano_data["tweet_text"].apply(safe_unicode_decode)
ano_data["tweet_text"] = ano_data["tweet_text"].apply(demojize)
ano_data["tweet_text"] = ano_data["tweet_text"].apply(normalizeTweet)

# print(ano_data.loc[58, 'tweet_text'])


# 步骤1：提取指定列（包含tweet_id用于去重）
selected_columns = ["tweet_id", "tweet_text", "image", "label"]
df_selected = ano_data[selected_columns]

# 步骤2：按tweet_id去重（保留第一条记录）
df_unique = df_selected.drop_duplicates(subset=["tweet_id"], keep="first")

# 步骤3：移除临时使用的tweet_id列（若需要保留则跳过这步）
df_final = df_unique.drop(columns=["tweet_id"])

# 步骤4：保存结果到新CSV（保留列名）
df_final.to_csv(f"{root_}/all_data.csv", index=False, sep='\t')
# print(df_final.loc[119, 'tweet_text'])

