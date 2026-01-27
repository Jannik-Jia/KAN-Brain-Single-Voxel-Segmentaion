
#!/usr/bin/env python3
"""
Load saved fold models (e.g. exclude-all runs), reproduce the original settings,
run predictions on the matching train/test split, and export per-class gross
accuracy/F1 together with a generalization coefficient (GC) check.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_1d_with_3d_dataset import RegModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved models with per-class metrics and GC validation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a single checkpoint (.pth).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory containing multiple checkpoints (evaluates all *.pth).",
    )
    parser.add_argument(
        "--data_dir_1d",
        type=str,
        default=None,
        help="Override 1D data directory (uses checkpoint args if omitted).",
    )
    parser.add_argument(
        "--data_dir_3d",
        type=str,
        default=None,
        help="Override 3D data directory (uses checkpoint args if omitted).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save per-class metric JSON files (defaults to checkpoint parent).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override evaluation batch size (defaults to checkpoint batch_size).",
    )
    parser.add_argument(
        "--gc_threshold",
        type=float,
        default=40.0,
        help="GC percentage threshold for triggering a fallback re-evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    return parser.parse_args()


def infer_input_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer input dimension from the first linear layer weight shape."""
    fc1_keys = [k for k in state_dict.keys() if k.endswith("fc1.weight")]
    if not fc1_keys:
        raise ValueError("Cannot find fc1.weight in checkpoint to infer input_dim.")
    fc1_weight = state_dict[fc1_keys[0]]
    return int(fc1_weight.shape[1])


class FlexibleBrain1DDataset(Dataset):
    """
    1D dataset loader with configurable input_dim (uses scaler if provided).
    Mirrors Brain1D_Dataset but allows arbitrary input_dim inferred from checkpoint.
    """

    def __init__(
        self,
        mat_files: list,
        is_train: bool,
        scaler: Optional[StandardScaler],
        input_dim: int,
    ):
        self.mat_files = mat_files
        self.is_train = is_train
        self.scaler = scaler
        self.input_dim = input_dim

        self.all_data = []
        self.all_labels = []

        for mat_file in mat_files:
            with h5py.File(mat_file, "r") as f:
                multidim_data = f["multidim_data"][()]
                seg_one_hot = f["seg_one_hot"][()]

                if multidim_data.shape[0] == 351:
                    multidim_data = multidim_data.T
                if seg_one_hot.shape[0] == 102:
                    seg_one_hot = seg_one_hot.T

                if multidim_data.shape[1] < input_dim:
                    raise ValueError(
                        f"{mat_file} has only {multidim_data.shape[1]} features, "
                        f"but model expects {input_dim}"
                    )

                multidim_data = multidim_data[:, :input_dim]
                labels = np.argmax(seg_one_hot, axis=1)

                self.all_data.append(multidim_data.astype(np.float32))
                self.all_labels.append(labels.astype(np.int64))

        self.all_data = np.vstack(self.all_data)
        self.all_labels = np.concatenate(self.all_labels)

        if self.scaler is None:
            if not is_train:
                raise ValueError("Scaler is required for eval; checkpoint should contain it.")
            self.scaler = StandardScaler()
            self.all_data = self.scaler.fit_transform(self.all_data).astype(np.float32)
        else:
            if hasattr(self.scaler, "mean_") and self.scaler.mean_.shape[0] != input_dim:
                raise ValueError(
                    f"Scaler feature dim {self.scaler.mean_.shape[0]} != input_dim {input_dim}"
                )
            self.all_data = self.scaler.transform(self.all_data).astype(np.float32)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx], self.all_labels[idx]


class FlexibleTestDataset(Dataset):
    """
    Test dataset with configurable input_dim; loads 1D features and 3D labels for alignment.
    """

    def __init__(
        self,
        mat_file_1d: Path,
        mat_file_3d: Path,
        scaler: StandardScaler,
        input_dim: int,
    ):
        if scaler is None:
            raise ValueError("Scaler is required for test dataset.")

        with h5py.File(mat_file_1d, "r") as f:
            multidim_data = f["multidim_data"][()]
            seg_one_hot = f["seg_one_hot"][()]

            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T
            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T

            if multidim_data.shape[1] < input_dim:
                raise ValueError(
                    f"{mat_file_1d} has only {multidim_data.shape[1]} features, "
                    f"but model expects {input_dim}"
                )

            multidim_data = multidim_data[:, :input_dim]
            self.features = multidim_data.astype(np.float32)
            self.labels = np.argmax(seg_one_hot, axis=1).astype(np.int64)

        with h5py.File(mat_file_3d, "r") as f:
            region_mask = f["region_mask"][()]
            region_labels = f["region_labels"][()]

            assert region_mask.shape == (384, 336, 256), (
                f"region_mask shape {region_mask.shape} invalid for {mat_file_3d}"
            )
            assert region_labels.shape == (384, 336, 256), (
                f"region_labels shape {region_labels.shape} invalid for {mat_file_3d}"
            )

            self.region_mask = region_mask
            self.region_labels = region_labels

        if hasattr(scaler, "mean_") and scaler.mean_.shape[0] != input_dim:
            raise ValueError(
                f"Scaler feature dim {scaler.mean_.shape[0]} != input_dim {input_dim}"
            )
        self.features = scaler.transform(self.features).astype(np.float32)

        # Label alignment check
        mask = self.region_mask.astype(bool)
        labels_3d = self.region_labels[mask]
        if len(labels_3d) != len(self.labels):
            raise ValueError(
                f"Label count mismatch: 1D={len(self.labels)} vs 3D={len(labels_3d)}"
            )
        if not np.array_equal(labels_3d, self.labels):
            mismatch = np.sum(labels_3d != self.labels)
            raise AssertionError(
                f"1D/3D labels mismatch: {mismatch}/{len(self.labels)} voxels differ"
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def build_split_files(
    data_dir_1d: Path, data_dir_3d: Path, test_subject: int
) -> Tuple[list, Path, Path, str, list]:
    mat_files_1d = list(data_dir_1d.glob("*.mat"))
    mat_files_3d = list(data_dir_3d.glob("*_3d_validated.mat"))

    if not mat_files_1d:
        raise FileNotFoundError(f"No 1D MAT files found in {data_dir_1d}")
    if not mat_files_3d:
        raise FileNotFoundError(f"No 3D MAT files found in {data_dir_3d}")

    def subject_key_1d(p: Path) -> str:
        return p.stem

    def subject_key_3d(p: Path) -> str:
        return p.stem.replace("_3d_validated", "")

    idx_1d = {subject_key_1d(p): p for p in mat_files_1d}
    idx_3d = {subject_key_3d(p): p for p in mat_files_3d}

    if set(idx_1d.keys()) != set(idx_3d.keys()):
        diff = idx_1d.keys() ^ idx_3d.keys()
        raise ValueError(f"1D/3D subject sets differ: {diff}")

    subject_names = sorted(idx_1d.keys())
    test_idx = test_subject - 1
    if test_idx < 0 or test_idx >= len(subject_names):
        raise ValueError(f"test_subject {test_subject} out of range 1-{len(subject_names)}")

    test_subject_name = subject_names[test_idx]
    test_file_1d = idx_1d[test_subject_name]
    test_file_3d = idx_3d[test_subject_name]

    train_subject_names = [n for n in subject_names if n != test_subject_name]
    train_files_1d = [idx_1d[n] for n in train_subject_names]

    return train_files_1d, test_file_1d, test_file_3d, test_subject_name, subject_names


def compute_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 102
) -> Tuple[Dict[str, Dict[str, float]], float, float, float]:
    per_class_correct = np.bincount(y_true[y_true == y_pred], minlength=n_classes)
    per_class_total = np.bincount(y_true, minlength=n_classes)

    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        zero_division=0,
        average=None,
    )

    per_class = {}
    for cls in range(n_classes):
        per_class[str(cls)] = {
            "support": int(per_class_total[cls]),
            "gross_accuracy": float(per_class_acc[cls]),
            "precision": float(precision[cls]),
            "recall": float(recall[cls]),
            "f1": float(f1[cls]),
        }

    macro_f1 = float(np.mean(f1))
    macro_gross_accuracy = float(np.mean(per_class_acc))
    overall_top1 = float((y_true == y_pred).mean())

    return per_class, macro_f1, macro_gross_accuracy, overall_top1


def compute_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """计算负对数似然损失 (Negative Log-Likelihood)"""
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    nll = -np.mean(np.log(y_prob_clipped[np.arange(len(y_true)), y_true]))
    return float(nll)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """计算多类 Brier Score"""
    n_classes = y_prob.shape[1]
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true)), y_true] = 1
    brier = np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))
    return float(brier)


def compute_soft_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """计算 Soft ECE (Expected Calibration Error)"""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece)


def compute_3d_soft_dice_macro(
    y_prob: np.ndarray,
    region_mask: np.ndarray,
    region_labels: np.ndarray,
    n_classes: int = 102,
) -> float:
    """
    计算 3D Soft Dice (Macro)
    将 1D 概率映射回 3D 空间，计算每类的 Soft Dice 并取宏平均
    """
    # 获取 mask 中有效体素的坐标
    mask_coords = np.argwhere(region_mask)
    n_voxels = mask_coords.shape[0]

    if n_voxels != y_prob.shape[0]:
        raise ValueError(
            f"Voxel count mismatch: mask has {n_voxels}, y_prob has {y_prob.shape[0]}"
        )

    # 计算每个类的 Soft Dice
    dice_per_class = []
    for c in range(n_classes):
        # 真实：该类的二值掩码（只在 region_mask 内的体素）
        true_c = (region_labels[region_mask.astype(bool)] == c).astype(np.float32)
        # 预测：该类的概率
        pred_c = y_prob[:, c]

        intersection = np.sum(pred_c * true_c)
        union = np.sum(pred_c) + np.sum(true_c)

        if union > 0:
            dice = 2 * intersection / union
        else:
            dice = 1.0  # 该类不存在时，认为完美匹配
        dice_per_class.append(dice)

    return float(np.mean(dice_per_class))


def compute_selective_prediction(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    coverage_target: float = 0.95,
    risk_target: float = 0.05,
) -> Dict[str, float]:
    """
    计算选择性预测指标

    排序依据: Max Softmax Probability (置信度)

    返回:
    - risk_hard_at_coverage: 在给定覆盖率下的硬风险（分类错误率 = 1 - Accuracy）
    - risk_soft_at_coverage: 在给定覆盖率下的软风险（1 - 真实类别的预测概率）
    - coverage_at_hard_risk: 在给定硬风险阈值下的最大覆盖率
    - coverage_at_soft_risk: 在给定软风险阈值下的最大覆盖率
    """
    # 置信度：Max Softmax Probability
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct = (predictions == y_true)

    # 真实类别的预测概率
    true_class_probs = y_prob[np.arange(len(y_true)), y_true]

    # 按置信度降序排序
    sorted_indices = np.argsort(-confidences)

    # Risk @95% Coverage
    n_select = int(len(y_true) * coverage_target)
    if n_select == 0:
        n_select = 1
    selected_indices = sorted_indices[:n_select]

    # Risk (Hard): 分类错误率 = 1 - Accuracy
    risk_hard_at_coverage = 1 - np.mean(correct[selected_indices])

    # Risk (Soft): 1 - 真实类别的预测概率
    risk_soft_at_coverage = 1 - np.mean(true_class_probs[selected_indices])

    # Coverage @5% Risk (Hard)
    coverage_at_hard_risk = 0.0
    for i in range(1, len(y_true) + 1):
        selected = sorted_indices[:i]
        current_risk = 1 - np.mean(correct[selected])
        if current_risk <= risk_target:
            coverage_at_hard_risk = i / len(y_true)

    # Coverage @5% Risk (Soft)
    coverage_at_soft_risk = 0.0
    for i in range(1, len(y_true) + 1):
        selected = sorted_indices[:i]
        current_risk = 1 - np.mean(true_class_probs[selected])
        if current_risk <= risk_target:
            coverage_at_soft_risk = i / len(y_true)

    return {
        "risk_hard_at_95cov": float(risk_hard_at_coverage),
        "risk_soft_at_95cov": float(risk_soft_at_coverage),
        "coverage_at_5pct_risk_hard": float(coverage_at_hard_risk),
        "coverage_at_5pct_risk_soft": float(coverage_at_soft_risk),
    }


def evaluate_split(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
    return_probs: bool = True,
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_preds: list = []
    all_labels: list = []
    all_probs: list = []

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(loader, desc=desc):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())
            if return_probs:
                all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs) if return_probs else None

    per_class, macro_f1, macro_gross_acc, overall_top1 = compute_per_class_metrics(y_true, y_pred)

    # 计算 Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    result = {
        "samples": int(len(y_true)),
        "overall_top1_accuracy": overall_top1,
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": macro_f1,
        "macro_gross_accuracy": macro_gross_acc,
        "per_class": per_class,
    }

    # 概率相关指标
    if y_prob is not None:
        result["nll"] = compute_nll(y_true, y_prob)
        result["brier_score"] = compute_brier_score(y_true, y_prob)
        result["soft_ece"] = compute_soft_ece(y_true, y_prob)

        # 选择性预测指标
        selective_metrics = compute_selective_prediction(
            y_true, y_prob, coverage_target=0.95, risk_target=0.05
        )
        result.update(selective_metrics)

        # 保存概率用于后续 3D Dice 计算
        result["_y_prob"] = y_prob
        result["_y_true"] = y_true

    return result


def compute_gc(train_metrics: Dict[str, Any], test_metrics: Dict[str, Any]) -> Dict[str, Any]:
    train_macro = train_metrics["macro_f1"]
    test_macro = test_metrics["macro_f1"]
    macro_gc_pct = 0.0 if train_macro <= 0 else 100.0 * test_macro / train_macro

    gc_per_class = {}
    for cls, cls_test in test_metrics["per_class"].items():
        train_f1 = train_metrics["per_class"][cls]["f1"]
        gc_per_class[cls] = None if train_f1 <= 0 else 100.0 * cls_test["f1"] / train_f1

    return {
        "overall_macro_f1_pct": macro_gc_pct,
        "per_class_f1_pct": gc_per_class,
    }


def run_full_evaluation(
    ckpt_path: Path,
    data_dir_1d: Path,
    data_dir_3d: Path,
    batch_size: int,
    gc_threshold: float,
    num_workers: int,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    test_subject = int(saved_args.get("test_subject", 1))
    state_dict = checkpoint["model_state_dict"]
    input_dim = infer_input_dim(state_dict)

    scaler = checkpoint.get("scaler")
    if scaler is None:
        raise ValueError(f"Checkpoint {ckpt_path} does not contain a scaler.")

    model = RegModel(input_dim=input_dim, num_classes=102)
    model.load_state_dict(state_dict)
    model = model.to(device)

    train_files_1d, test_file_1d, test_file_3d, test_subject_name, subject_names = build_split_files(
        data_dir_1d, data_dir_3d, test_subject
    )

    train_dataset = FlexibleBrain1DDataset(
        train_files_1d,
        is_train=False,
        scaler=scaler,
        input_dim=input_dim,
    )
    test_dataset = FlexibleTestDataset(
        test_file_1d,
        test_file_3d,
        scaler,
        input_dim=input_dim,
    )

    train_metrics = evaluate_split(
        model,
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        desc="Eval-train",
        return_probs=True,
    )
    test_metrics = evaluate_split(
        model,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        desc="Eval-test",
        return_probs=True,
    )

    # 计算测试集的 3D Soft Dice (需要 3D 数据)
    if "_y_prob" in test_metrics:
        test_3d_soft_dice = compute_3d_soft_dice_macro(
            test_metrics["_y_prob"],
            test_dataset.region_mask,
            test_dataset.region_labels,
            n_classes=102,
        )
        test_metrics["3d_soft_dice_macro"] = test_3d_soft_dice

    gc_metrics = compute_gc(train_metrics, test_metrics)

    # 清理内部数据，不保存到 JSON
    for key in ["_y_prob", "_y_true"]:
        train_metrics.pop(key, None)
        test_metrics.pop(key, None)

    return {
        "checkpoint": str(ckpt_path),
        "test_subject": test_subject,
        "test_subject_name": test_subject_name,
        "all_subjects": subject_names,
        "data_dir_1d": str(data_dir_1d),
        "data_dir_3d": str(data_dir_3d),
        "batch_size": batch_size,
        "input_dim": input_dim,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gc": gc_metrics,
        "gc_threshold": gc_threshold,
    }


def evaluate_with_fallback(
    ckpt_path: Path,
    args: argparse.Namespace,
    saved_args: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    data_dir_1d_primary = Path(args.data_dir_1d or saved_args.get("data_dir_1d"))
    data_dir_3d_primary = Path(args.data_dir_3d or saved_args.get("data_dir_3d"))

    if not data_dir_1d_primary.exists() or not data_dir_3d_primary.exists():
        raise FileNotFoundError("Provided data directories do not exist.")

    batch_size = args.batch_size or int(saved_args.get("batch_size", 128))

    primary = run_full_evaluation(
        ckpt_path,
        data_dir_1d_primary,
        data_dir_3d_primary,
        batch_size,
        args.gc_threshold,
        args.num_workers,
        device,
    )

    passes = {"initial": primary}
    final_key = "initial"

    if primary["gc"]["overall_macro_f1_pct"] < args.gc_threshold:
        print(
            f"GC below {args.gc_threshold}% for {ckpt_path.name}; re-evaluating with original data dirs."
        )
        data_dir_1d_fallback = Path(saved_args.get("data_dir_1d"))
        data_dir_3d_fallback = Path(saved_args.get("data_dir_3d"))

        fallback = run_full_evaluation(
            ckpt_path,
            data_dir_1d_fallback,
            data_dir_3d_fallback,
            batch_size,
            args.gc_threshold,
            args.num_workers,
            device,
        )
        passes["fallback_original_data"] = fallback
        final_key = "fallback_original_data"

    return {
        "passes": passes,
        "final_pass": final_key,
    }


def save_metrics(output_dir: Path, metrics: Dict[str, Any], test_subject: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"per_class_metrics_test{test_subject}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_path


def print_summary_table(all_results: list) -> None:
    """打印汇总表格，计算所有 Fold/Subject 的 Mean 和 Std"""
    if not all_results:
        return

    # 定义要汇总的指标
    test_metrics_keys = [
        ("overall_top1_accuracy", "test_gross_acc"),
        ("balanced_accuracy", "test_balanced_acc"),
        ("macro_f1", "test_macro_f1"),
        ("nll", "test_nll"),
        ("brier_score", "test_brier"),
        ("soft_ece", "test_soft_ece"),
        ("3d_soft_dice_macro", "test_3d_soft_dice_macro"),
        ("risk_hard_at_95cov", "test_risk_hard_at_95cov"),
        ("risk_soft_at_95cov", "test_risk_soft_at_95cov"),
        ("coverage_at_5pct_risk_hard", "test_coverage_at_5pct_risk_hard"),
        ("coverage_at_5pct_risk_soft", "test_coverage_at_5pct_risk_soft"),
    ]

    train_metrics_keys = [
        ("overall_top1_accuracy", "train_gross_acc"),
        ("balanced_accuracy", "train_balanced_acc"),
        ("macro_f1", "train_macro_f1"),
    ]

    gc_keys = [
        ("overall_macro_f1_pct", "gc_macro_f1_pct"),
    ]

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (across all folds/subjects)")
    print("=" * 80)

    # 收集数据
    summary_data = {}

    for result in all_results:
        final_pass = result["passes"][result["final_pass"]]

        # Test metrics
        for key, name in test_metrics_keys:
            if key in final_pass["test_metrics"]:
                if name not in summary_data:
                    summary_data[name] = []
                summary_data[name].append(final_pass["test_metrics"][key])

        # Train metrics
        for key, name in train_metrics_keys:
            if key in final_pass["train_metrics"]:
                if name not in summary_data:
                    summary_data[name] = []
                summary_data[name].append(final_pass["train_metrics"][key])

        # GC metrics
        for key, name in gc_keys:
            if key in final_pass["gc"]:
                if name not in summary_data:
                    summary_data[name] = []
                summary_data[name].append(final_pass["gc"][key])

    # 打印表格
    print(f"\n{'Metric':<35} {'Mean':>12} {'Std':>12} {'N':>6}")
    print("-" * 70)

    for name, values in summary_data.items():
        values_arr = np.array(values)
        mean_val = np.mean(values_arr)
        std_val = np.std(values_arr)
        n = len(values)
        print(f"{name:<35} {mean_val:>12.4f} {std_val:>12.4f} {n:>6}")

    print("-" * 70)
    print()


def save_summary_csv(output_dir: Path, all_results: list) -> Path:
    """保存汇总 CSV 文件"""
    if not all_results:
        return None

    # 收集所有 fold 的数据
    rows = []
    for result in all_results:
        final_pass = result["passes"][result["final_pass"]]
        row = {
            "test_subject": final_pass["test_subject"],
            "test_subject_name": final_pass["test_subject_name"],
        }

        # Test metrics
        for key in ["overall_top1_accuracy", "balanced_accuracy", "macro_f1",
                    "nll", "brier_score", "soft_ece", "3d_soft_dice_macro",
                    "risk_hard_at_95cov", "risk_soft_at_95cov",
                    "coverage_at_5pct_risk_hard", "coverage_at_5pct_risk_soft"]:
            if key in final_pass["test_metrics"]:
                row[f"test_{key}"] = final_pass["test_metrics"][key]

        # Train metrics
        for key in ["overall_top1_accuracy", "balanced_accuracy", "macro_f1"]:
            if key in final_pass["train_metrics"]:
                row[f"train_{key}"] = final_pass["train_metrics"][key]

        # GC
        row["gc_macro_f1_pct"] = final_pass["gc"]["overall_macro_f1_pct"]

        rows.append(row)

    # 按 test_subject 排序
    rows.sort(key=lambda x: x["test_subject"])

    # 写入 CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary_metrics.csv"

    import csv
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

            # 添加 Mean 和 Std 行
            mean_row = {"test_subject": "Mean", "test_subject_name": ""}
            std_row = {"test_subject": "Std", "test_subject_name": ""}
            for key in fieldnames[2:]:  # 跳过 test_subject 和 test_subject_name
                values = [r[key] for r in rows if key in r and r[key] is not None]
                if values:
                    mean_row[key] = np.mean(values)
                    std_row[key] = np.std(values)
            writer.writerow(mean_row)
            writer.writerow(std_row)

    return csv_path


def main():
    args = parse_args()

    if not args.checkpoint and not args.checkpoint_dir:
        raise SystemExit("Provide --checkpoint or --checkpoint_dir.")

    if args.checkpoint_dir and args.checkpoint:
        raise SystemExit("Use either --checkpoint or --checkpoint_dir, not both.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.checkpoint_dir:
        ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.pth"))
    else:
        ckpt_paths = [Path(args.checkpoint)]

    print(f"Found {len(ckpt_paths)} checkpoint(s)")

    if not ckpt_paths:
        raise SystemExit("No checkpoints found to evaluate.")

    all_results = []

    for ckpt_path in ckpt_paths:
        print(f"\nLoading {ckpt_path.name}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        saved_args = checkpoint.get("args", {})
        if not saved_args:
            print(f"  Skipping (missing args).")
            continue

        print(f"  test_subject: {saved_args.get('test_subject')}")
        print(f"  data_dir_1d: {saved_args.get('data_dir_1d')}")
        print(f"  data_dir_3d: {saved_args.get('data_dir_3d')}")
        print("  Starting evaluation...")

        result = evaluate_with_fallback(ckpt_path, args, saved_args, device)
        all_results.append(result)

        final_pass = result["passes"][result["final_pass"]]
        output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
        out_path = save_metrics(output_dir, result, final_pass["test_subject"])
        print(f"Saved per-class metrics to {out_path}")

    # 打印汇总统计
    print_summary_table(all_results)

    # 保存汇总 CSV
    if len(all_results) > 1:
        output_dir = Path(args.output_dir) if args.output_dir else ckpt_paths[0].parent
        csv_path = save_summary_csv(output_dir, all_results)
        if csv_path:
            print(f"Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
