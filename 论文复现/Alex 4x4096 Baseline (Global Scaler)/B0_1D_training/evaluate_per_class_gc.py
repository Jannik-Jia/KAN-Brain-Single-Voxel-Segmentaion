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

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_1d_with_3d_dataset import RegModel, Brain1D_Dataset, TestDataset


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


def evaluate_split(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
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

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(loader, desc=desc):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    per_class, macro_f1, macro_gross_acc, overall_top1 = compute_per_class_metrics(y_true, y_pred)

    return {
        "samples": int(len(y_true)),
        "overall_top1_accuracy": overall_top1,
        "macro_f1": macro_f1,
        "macro_gross_accuracy": macro_gross_acc,
        "per_class": per_class,
    }


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

    scaler = checkpoint.get("scaler")
    if scaler is None:
        raise ValueError(f"Checkpoint {ckpt_path} does not contain a scaler.")

    model = RegModel(input_dim=341, num_classes=102)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    train_files_1d, test_file_1d, test_file_3d, test_subject_name, subject_names = build_split_files(
        data_dir_1d, data_dir_3d, test_subject
    )

    train_dataset = Brain1D_Dataset(train_files_1d, is_train=False, scaler=scaler)
    test_dataset = TestDataset(test_file_1d, test_file_3d, scaler)

    train_metrics = evaluate_split(
        model,
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        desc="Eval-train",
    )
    test_metrics = evaluate_split(
        model,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        desc="Eval-test",
    )

    gc_metrics = compute_gc(train_metrics, test_metrics)

    return {
        "checkpoint": str(ckpt_path),
        "test_subject": test_subject,
        "test_subject_name": test_subject_name,
        "all_subjects": subject_names,
        "data_dir_1d": str(data_dir_1d),
        "data_dir_3d": str(data_dir_3d),
        "batch_size": batch_size,
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

    if not ckpt_paths:
        raise SystemExit("No checkpoints found to evaluate.")

    for ckpt_path in ckpt_paths:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        saved_args = checkpoint.get("args", {})
        if not saved_args:
            print(f"Skipping {ckpt_path} (missing args).")
            continue

        result = evaluate_with_fallback(ckpt_path, args, saved_args, device)

        final_pass = result["passes"][result["final_pass"]]
        output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
        out_path = save_metrics(output_dir, result, final_pass["test_subject"])
        print(f"Saved per-class metrics to {out_path}")


if __name__ == "__main__":
    main()
