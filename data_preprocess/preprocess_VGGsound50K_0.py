import pandas as pd
import os

filepath = 'E:/Gray/Database/VGGSound'
df = pd.read_csv('E:/Gray/Database/VGGSound-AVEL50k/vggsound-avel50k.csv')

VGGS50K_names = df['video_id']
VGGS50K_clss = df['category']

# 提取 'category' 列中的唯一元素
unique_categories = df['category'].unique()
# 将唯一元素转换为字典，键为元素，值为从0开始的类别序号
category_to_index = {category: idx for idx, category in enumerate(unique_categories)}
vid_to_cls = {VGGS50K_names[i]: VGGS50K_clss[i] for i in range(len(VGGS50K_names))}

file_list = []
vid_to_path = {}
for root, dirs, files in os.walk(filepath):
    for file in files:
        vid = file[1:12]
        file_list.append(vid)
        if not vid in vid_to_path.keys():
            vid_to_path.update({f'{vid}': f'{os.path.join(root, file)}'})

error = 0
with open('D:/Gray/Database/VGGS50K/VGGS50K_videos.txt', 'w') as file:
    for name in VGGS50K_names:
        if name in vid_to_path.keys():
            path = vid_to_path[name].replace('\\', '/')
            label = category_to_index[vid_to_cls[name]]
            file.write(f'{path}&{label}\n')
        else:
            error+=1
print(f'Error Video Number: {error}')

"""
{'engine accelerating': 0, 'playing trumpet': 1, 'race car': 2, 'orchestra': 3, 'lighting firecrackers': 4, 'playing violin': 5, 'playing erhu': 6, 'playing bass guitar': 7, 'playing snare drum': 8, 'cat purring': 9, 'playing harp': 10, 'people sniggering': 11, 'child singing': 12, 'goose honking': 13, 'ice cream truck': 14, 'playing bagpipes': 15, 'electric shaver': 16, 'people booing': 17, 'driving buses': 18, 'train horning': 19, 'police car (siren)': 20, 'wind noise': 21, 'playing clarinet': 22, 'people burping': 23, 'vehicle horn': 24, 'playing cymbal': 25, 'singing bowl': 26, 'playing badminton': 27, 'stream burbling': 28, 'cap gun shooting': 29, 'male singing': 30, 'vacuum cleaner cleaning floors': 31, 'rope skipping': 32, 'arc welding': 33, 'scuba diving': 34, 'playing bassoon': 35, 'people clapping': 36, 'playing harpsichord': 37, 'beat boxing': 38, 'playing double bass': 39, 'railroad car': 40, 'playing cello': 41, 'basketball bounce': 42, 'playing tabla': 43, 'civil defense siren': 44, 'pheasant crowing': 45, 'playing accordion': 46, 'gibbon howling': 47, 'playing drum kit': 48, 'people marching': 49, 'rowboat': 50, 'tractor digging': 51, 'dog barking': 52, 'toilet flushing': 53, 'cricket chirping': 54, 'playing french horn': 55, 'playing acoustic guitar': 56, 'playing banjo': 57, 'playing volleyball': 58, 'car engine knocking': 59, 'female singing': 60, 'playing mandolin': 61, 'bird chirping': 62, 'dog howling': 63, 'playing squash': 64, 'mynah bird singing': 65, 'machine gun shooting': 66, 'airplane flyby': 67, 'child speech': 68, 'missile launch': 69, 'fireworks banging': 70, 'ambulance siren': 71, 'playing marimba': 72, 'fire truck siren': 73, 'playing cornet': 74, 'pigeon': 75, 'skateboarding': 76, 'chainsawing trees': 77, 'people screaming': 78, 'people crowd': 79, 'skidding': 80, 'playing saxophone': 81, 'playing didgeridoo': 82, 'playing vibraphone': 83, 'playing bongo': 84, 'motorboat': 85, 'subway': 86, 'bowling impact': 87, 'playing piano': 88, 'dog growling': 89, 'lions roaring': 90, 'planing timber': 91, 'skiing': 92, 'lawn mowing': 93, 'playing electric guitar': 94, 'playing sitar': 95, 'lathe spinning': 96, 'playing bass drum': 97, 'typing on typewriter': 98, 'driving motorcycle': 99, 'sharpen knife': 100, 'people cheering': 101, 'ocean burbling': 102, 'church bell ringing': 103, 'singing choir': 104, 'playing electronic organ': 105, 'horse clip-clop': 106, 'people whistling': 107, 'playing glockenspiel': 108, 'people whispering': 109, 'male speech': 110, 'owl hooting': 111, 'frog croaking': 112, 'female speech': 113, 'playing tambourine': 114, 'playing table tennis': 115, 'printer printing': 116, 'roller coaster running': 117, 'crow cawing': 118, 'police radio chatter': 119, 'turkey gobbling': 120, 'tap dancing': 121, 'playing synthesizer': 122, 'helicopter': 123, 'playing hammond organ': 124, 'chicken crowing': 125, 'cattle': 126, 'playing steel guitar': 127, 'woodpecker pecking tree': 128, 'cattle mooing': 129, 'playing trombone': 130, 'playing flute': 131, 'playing ukulele': 132, 'volcano explosion': 133, 'canary calling': 134, 'baby laughter': 135, 'playing harmonica': 136, 'slot machine': 137, 'playing theremin': 138, 'yodelling': 139, 'tapping guitar': 140}
"""
