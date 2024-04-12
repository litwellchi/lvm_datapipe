import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


# 以下代码是关于动画片视频数据结构的
# vids_path = r'F:\youtube'   # TODO
# vids = os.listdir(vids_path)
# category_counts = {}
# durations = []
# for vid in tqdm(vids):
#     if vid.endswith('.json'):
#         json_path = os.path.join(vids_path, vid)
#         if not os.path.exists(json_path.replace('.info.json', '.mp4')):
#             continue
#         with open(json_path, 'r') as f:
#             json_data = json.load(f)
#         durations.append(json_data['duration'])
#         category = json_data['categories']
#         for cat in category:
#             if cat in category_counts:
#                 category_counts[cat] += 1
#             else:
#                 category_counts[cat] = 1

# For trailers 
jsons_path = r''   # TODO
jsons = os.listdir(jsons_path)
category_counts = {}
durations = []
for json_p in tqdm(jsons):
    if 'json' in json_p:
        json_path = os.path.join(jsons_path, json_p)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        for x in json_data:     # TODO: add whatever condition for filtering 
            durations.append(x['basic']['clip_duration'])


print(sum(durations), sum(durations)/len(durations))
my_bins = [0, 5, 10, 60, 300, 600, 1800, 3600, float('inf')]
hist, bin_edges = np.histogram(durations, bins=my_bins)
duration_counts = hist.tolist()

# 绘制饼图
plt.pie(duration_counts, labels=bin_edges[:-1], autopct='%1.1f%%')
plt.title("Youtube-1.5T Video Duration Distribution")
plt.savefig('vid_duration_distribution.png')
plt.clf()

# plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
# plt.title("Youtube-1.5T Category Distribution")
# plt.savefig('vid_category_distribution.png')


