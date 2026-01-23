#!/usr/bin/env python3
"""
从“排除全部”实验的Checkpoint恢复模型，并在同一数据集上重新预测以获得宏平均F1（macro F1）和
Gross Accuracy（GC）。

- 默认读取路径: results_exclude_experiments_full/exp7_exclude_all/dense_4x4096_model_test*.pth
- 严格复用训练时保存的参数（数据路径、排除列表、测试被试选择、batch size 等）
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from evaluation_metrics import compute_all_metrics
from train_1d_with_3d_dataset import (
    RegModel,
    TestDataset,
    load_exclude_list,
    should_exclude_subject,
)


def _find_default_checkpoint() -> Path:
    """在默认目录中查找第一个checkpoint文件"""
    exp_dir = Path("results_exclude_experiments_full/exp7_exclude_all")
    if exp_dir.exists():
        ckpts = sorted(exp_dir.glob("dense_4x4096_model_test*.pth"))
        if ckpts:
            return ckpts[0]
    return exp_dir / "dense_4x4096_model_test38.pth"


def _subject_key_1d(p: Path) -> str:
    return p.stem


def _subject_key_3d(p: Path) -> str:
    return p.stem.replace("_3d_validated", "")


def build_test_loader(saved_args: dict, batch_size: int) -> Tuple[DataLoader, str]:
    """
    使用训练时保存的参数构建测试集 DataLoader。
    返回 DataLoader 和实际使用的测试被试名。
    """
    data_dir_1d = Path(saved_args["data_dir_1d"])
    data_dir_3d = Path(saved_args["data_dir_3d"])

    mat_files_1d = list(data_dir_1d.glob("*.mat"))
    mat_files_3d = list(data_dir_3d.glob("*_3d_validated.mat"))

    idx_1d = {_subject_key_1d(p): p for p in mat_files_1d}
    idx_3d = {_subject_key_3d(p): p for p in mat_files_3d}
    if set(idx_1d.keys()) != set(idx_3d.keys()):
        raise ValueError("一维和三维被试集合不一致，无法复现原始测试集选择逻辑")

    # 组装排除集
    exclude_set = set()
    if saved_args.get("exclude_subjects"):
        exclude_file = Path(saved_args["exclude_subjects"])
        exclude_set = load_exclude_list(exclude_file)
    if saved_args.get("exclude_single"):
        exclude_set.add(saved_args["exclude_single"])

    subject_names = sorted(
        [name for name in idx_1d.keys() if not should_exclude_subject(name, exclude_set)]
    )

    # 确定测试被试
    if saved_args.get("fixed_test_subject"):
        fixed = saved_args["fixed_test_subject"]
        test_subject_name = next((n for n in subject_names if fixed in n), None)
        if test_subject_name is None:
            raise ValueError(f"未找到匹配的固定测试被试: {fixed}")
    else:
        test_idx = saved_args.get("test_subject", 1) - 1
        if test_idx >= len(subject_names):
            raise ValueError(
                f"测试被试编号{saved_args.get('test_subject')}超出范围[1, {len(subject_names)}]"
            )
        test_subject_name = subject_names[test_idx]

    test_file_1d = idx_1d[test_subject_name]
    test_file_3d = idx_3d[test_subject_name]

    test_dataset = TestDataset(test_file_1d, test_file_3d)

    # 标签一致性自检（与训练脚本保持一致的安全检查）
    mask = test_dataset.region_mask.astype(bool)
    labels_3d = test_dataset.region_labels[mask]
    labels_1d = test_dataset.labels
    if len(labels_1d) != len(labels_3d) or not np.array_equal(labels_1d, labels_3d):
        raise AssertionError("测试集标签在一维与三维不一致，无法复现原始评估")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader, test_subject_name


def collect_predictions(model: RegModel, loader: DataLoader, device: torch.device):
    """单次遍历收集预测结果，返回 y_true, y_pred, y_probs"""
    model.eval()
    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)

            all_true.extend(target.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_probs = np.array(all_probs)
    return y_true, y_pred, y_probs


def main():
    parser = argparse.ArgumentParser(description="评估排除全部实验的Checkpoint (Macro F1 & GC)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_find_default_checkpoint()),
        help="模型checkpoint路径（默认指向 exp7_exclude_all）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="可选：保存评估指标的JSON路径（默认与checkpoint同目录下的 eval_metrics.json）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="可选：指定设备，如 cuda 或 cpu（默认自动检测cuda）",
    )
    parser.add_argument(
        "--batch_size_override",
        type=int,
        default=None,
        help="可选：覆盖checkpoint中保存的batch size",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})

    # 使用保存的batch_size，除非显式覆盖
    batch_size = args.batch_size_override or saved_args.get("batch_size", 128)

    # 设置随机种子与训练保持一致
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 设备设置
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 构建测试集
    test_loader, test_subject_name = build_test_loader(saved_args, batch_size)

    # 构建并加载模型
    model = RegModel(input_dim=351, num_classes=102)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 评估：单次前向收集，再计算整体与逐类指标
    y_true, y_pred, y_probs = collect_predictions(model, test_loader, device)
    metrics = compute_all_metrics(y_true, y_pred, y_probs)

    n_classes = y_probs.shape[1]
    per_class_acc = []
    per_class_f1 = f1_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
    ).tolist()


    for c in range(n_classes):
        mask = (y_true == c)
        if np.sum(mask) == 0:
            per_class_acc.append(None)  # JSON-safe: null
        else:
            per_class_acc.append(float(np.mean(y_pred[mask] == y_true[mask])))

    result = {
        "checkpoint": str(ckpt_path),
        "test_subject_name": test_subject_name,
        "batch_size": batch_size,
        "gross_accuracy": metrics.get("gross_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "top1_accuracy": metrics.get("top1_accuracy"),
        "top3_accuracy": metrics.get("top3_accuracy"),
        "top5_accuracy": metrics.get("top5_accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "weighted_f1": metrics.get("weighted_f1"),
        "per_class_gross_accuracy": per_class_acc,
        "per_class_f1": per_class_f1,
    }

    # 打印核心指标
    print("\n===== Exclude-All Checkpoint Evaluation =====")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test subject: {test_subject_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Gross Accuracy: {result['gross_accuracy']:.4f}")
    print(f"Macro F1: {result['macro_f1']:.4f}")

    # 保存结果
    output_json = (
        Path(args.output_json)
        if args.output_json
        else ckpt_path.parent / "eval_metrics_exclude_all.json"
    )

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)

    print(f"指标已保存到: {output_json}")


if __name__ == "__main__":
    main()
