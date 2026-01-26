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
from sklearn.metrics import precision_recall_fscore_support
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

        final_pass = result["passes"][result["final_pass"]]
        output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
        out_path = save_metrics(output_dir, result, final_pass["test_subject"])
        print(f"Saved per-class metrics to {out_path}")


if __name__ == "__main__":
    main()
