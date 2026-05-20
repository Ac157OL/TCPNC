import os
import numpy as np
import tifffile
from cellpose4 import metrics
import tools


def evaluate_dataset(pred_root, gt_root):
    dset_name = os.path.basename(os.path.normpath(pred_root))
    save_root = os.path.dirname(os.path.normpath(pred_root))

    # 读取 GT 文件列表（只取文件名）
    gt_files = [f for f in os.listdir(gt_root) if f.endswith(".json")]

    for model_name in os.listdir(pred_root):
        model_dir = os.path.join(pred_root, model_name)
        if not os.path.isdir(model_dir):
            continue

        print(f"\nProcessing model: {model_name}")

        masks_true = []
        masks_pred = []

        # 获取当前模型文件夹下所有预测文件
        pred_files = [f for f in os.listdir(model_dir) if f.lower().endswith((".json", ".tif", ".tiff"))]

        processed_filenames = []
        for pred_file in pred_files:
            pred_name = os.path.splitext(pred_file)[0]

            # --- 核心简化匹配逻辑 ---
            # 只要 gt_files 中的某个文件名字符串包含在 pred_name 中，就返回该文件名
            matched_gt = next((f for f in gt_files if os.path.splitext(f)[0] in pred_name), None)

            if matched_gt is None:
                continue
            # -----------------------

            # 读取数据
            gt_path = os.path.join(gt_root, matched_gt)
            gt_mask = tools.json_to_mask(gt_path)

            pred_path = os.path.join(model_dir, pred_file)
            if pred_file.lower().endswith(".json"):
                pred_mask = tools.json_to_mask(pred_path)
            else:
                pred_mask = tifffile.imread(pred_path)

            # 过滤逻辑
            pred_mask = tools.filter_pred_by_gt_overlap_ratio(pred_mask, gt_mask, overlap_ratio_threshold=0.3)

            masks_true.append(gt_mask)
            masks_pred.append(pred_mask)
            processed_filenames.append(pred_file)

        if len(masks_true) == 0:
            print(f"No matched files for {model_name}")
            continue

        threshold = np.array([0.5, 0.75, 0.9])
        ap, tp, fp, fn = metrics.average_precision(
            masks_true,
            masks_pred,
            threshold=threshold
        )

        save_path = os.path.join(
            save_root,
            f"{model_name}_{dset_name}.npy"
        )

        np.save(save_path, {
            "ap": ap,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "threshold": threshold,
            "filenames": processed_filenames,
            "masks_true": masks_true,
            "masks_pred": masks_pred
        })

        print(f"Saved to {save_path}")


if __name__ == "__main__":
    pred_root = r"D:\lab\cell\LatestResultV3.0\sansha-5"  # sansha-5 WO115-2
    gt_path = r"D:\lab\cell\LatestDataset\sansha-5"

    evaluate_dataset(pred_root, gt_path)
