import os
import numpy as np
from skimage.io import imread
from cellpose4 import utils
import tools
from cellpose4 import plot
# ========================
# 配置
# ========================
dataset_names = ["WO115-2", "sansha-5"]
gt_root = r"D:\lab\cell\LatestDataset"


def load_mask_file(file_path):
    """读取 json 或 mask 图像"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".json":
        return tools.json_to_mask(file_path)

    elif ext in [".png", ".tif", ".tiff", ".jpg"]:
        mask = imread(file_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return np.maximum(mask, 0).astype(np.int32)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ========================
# 计算平均直径
# ========================
for dataset_name in dataset_names:

    gt_dataset_dir = os.path.join(gt_root, dataset_name)

    if not os.path.isdir(gt_dataset_dir):
        print(f"[SKIP] {dataset_name} not found")
        continue

    print(f"\nDataset: {dataset_name}")

    gt_files = [f for f in os.listdir(gt_dataset_dir) if f.lower().endswith(".json")]
    gt_files.sort()

    all_diams = []

    for gt_file in gt_files:

        gt_path = os.path.join(gt_dataset_dir, gt_file)

        masks_gt = load_mask_file(gt_path)

        # 计算该图像的细胞直径
        diam = utils.diameters(masks_gt)[0]

        all_diams.append(diam)

    if len(all_diams) > 0:

        avg_diam = np.mean(all_diams)
        median_diam = np.median(all_diams)

        print(f"Images: {len(all_diams)}")
        print(f"Mean cell diameter : {avg_diam:.2f} pixels")
        print(f"Median diameter    : {median_diam:.2f} pixels")

    else:
        print("No GT masks found.")