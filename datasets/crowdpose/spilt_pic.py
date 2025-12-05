import os
import shutil
from tqdm import tqdm

# ====== 配置区 ======
dataset_root = 'datasets/crowdpose'  # 根目录
image_src_dir = os.path.join(dataset_root, 'images')  # 原始图像文件夹
label_root = os.path.join(dataset_root, 'labels')     # 已经分好的标签文件夹
# ====================

# 遍历标签子集
for subset in ['train', 'val', 'test']:
    label_dir = os.path.join(label_root, subset)
    image_dst_dir = os.path.join(dataset_root, 'images', subset)
    os.makedirs(image_dst_dir, exist_ok=True)

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    print(f"\n📂 正在复制 {subset} 图片，共 {len(label_files)} 张...")
    for label_file in tqdm(label_files):
        img_name = os.path.splitext(label_file)[0] + '.jpg'
        src_img = os.path.join(image_src_dir, img_name)
        dst_img = os.path.join(image_dst_dir, img_name)

        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f"⚠️ 找不到图片：{img_name}")
