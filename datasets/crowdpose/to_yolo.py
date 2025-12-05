import json
import os

json_file = 'crowdpose_test.json'
output_dir = 'labels/test'
class_id = 0

os.makedirs(output_dir, exist_ok=True)

with open(json_file, 'r') as f:
    data = json.load(f)

# 建立 image_id -> (file_name, width, height)
image_map = {img['id']: (img['file_name'], img['width'], img['height']) for img in data['images']}

for ann in data['annotations']:
    image_id = ann['image_id']
    bbox = ann['bbox']
    x_min, y_min, width, height = bbox

    # 跳过非法框
    if width <= 0 or height <= 0:
        continue

    file_name, img_width, img_height = image_map[image_id]
    txt_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.txt')
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)

    # 归一化
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    w = width / img_width
    h = height / img_height

    # 若包含关键点（pose任务）
    if 'keypoints' in ann:
        kpts = ann['keypoints']
        kpt_str = ''
        for i in range(0, len(kpts), 3):
            x = kpts[i] / img_width
            y = kpts[i + 1] / img_height
            v = kpts[i + 2]
            kpt_str += f" {x:.6f} {y:.6f} {v}"
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}{kpt_str}\n"
    else:
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

    with open(txt_file, 'a') as f:
        f.write(line)
