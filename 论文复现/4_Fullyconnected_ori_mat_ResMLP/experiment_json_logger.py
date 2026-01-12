#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment JSON Logger for MLP experiments on Alex7T dataset.
Integrates with the unified experiment schema from 通用工具.

This module provides:
1. ExperimentJSONLogger: Main class for creating and saving experiment JSON records
2. Integration with existing training code
3. Automatic metric calculation using MetricCalculator

Usage:
    from experiment_json_logger import ExperimentJSONLogger

    # Initialize at start of training
    logger = ExperimentJSONLogger(
        experiment_id="alex_deepmlp_6x2048_patientwise_v1",
        method_name="Deep MLP 6x2048 (BayesOpt best)",
        method_key="deepmlp_6x2048_patientwise",
        family="mlp",
        subfamily="deepmlp_6x2048"
    )

    # Configure from training config
    logger.set_from_config(config)

    # After training, log results
    logger.log_results(y_true, y_pred_proba, split="test")

    # Save
    logger.save(config['save_dir'])
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch

# Add 通用工具 to path if needed
COMMON_TOOLS_PATH = Path(__file__).parent.parent / "通用工具"
if str(COMMON_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS_PATH))

try:
    from metrics import MetricCalculator
    from validator import SchemaValidator
except ImportError:
    # Fallback: define minimal versions if import fails
    print("Warning: Could not import from 通用工具, using fallback implementations")
    MetricCalculator = None
    SchemaValidator = None


class ExperimentJSONLogger:
    """
    Experiment JSON Logger for MLP-based voxel classification experiments.

    Follows the unified experiment schema for Alex7T MPRAGE hard-label experiments.
    """

    # Short-term early stopping template (n_epochs ~ 30-40)
    SHORT_TERM_EARLY_STOPPING = {
        "enabled": True,
        "monitor": "val_macro_f1",
        "mode": "max",
        "min_epochs": 15,
        "patience": 5,
        "delta": 0.001
    }

    # Long-term early stopping template (n_epochs ~ 100, AttnResMLP, KAN)
    LONG_TERM_EARLY_STOPPING = {
        "enabled": True,
        "monitor": "val_macro_f1",
        "mode": "max",
        "min_epochs": 40,
        "patience": 10,
        "delta": 0.0005
    }

    def __init__(
        self,
        experiment_id: str,
        method_name: str,
        method_key: str,
        family: str,
        subfamily: str,
        seed: int = 666
    ):
        """
        Initialize the experiment logger.

        Args:
            experiment_id: Unique experiment identifier (e.g., "alex_deepmlp_6x2048_v1")
            method_name: Human-readable method name
            method_key: Short key for tables/figures
            family: Model family ("mlp", "kan", "tabnet", etc.)
            subfamily: More specific model type (e.g., "deepmlp_6x2048")
            seed: Random seed used
        """
        self.data = {
            "experiment_id": experiment_id,
            "method_name": method_name,
            "method_key": method_key,
            "family": family,
            "subfamily": subfamily,
            "code_version": self._get_git_hash(),
            "seed": seed,
            "timestamp": datetime.now().isoformat(),

            # Initialize empty structures
            "task": {
                "dataset": "alex",
                "label_space": "alex-101",  # Default for no-label-43 dataset
                "grid": "mprage",
                "split": {
                    "scheme": "subject-wise",
                    "split_id": "main_split",
                    "train_subjects": [],
                    "val_subjects": [],
                    "test_subjects": []
                },
                "n_voxels": {
                    "train": 0,
                    "val": 0,
                    "test": 0
                }
            },
            "preprocessing": {
                "feature_version": "voxel_signature_v1",
                "normalisation": {
                    "type": "per_patient_zscore",
                    "params": {
                        "epsilon": 1e-10,
                        "fit_scope": "per_subject"
                    }
                },
                "background_handling": "include_as_class_0",
                "class_weighting": {
                    "scheme": "inverse_frequency",
                    "computed_on": "train_foreground_voxels",
                    "weights_file": None
                }
            },
            "model": {
                "family": family,
                "param_count_m": None,
                "details": {
                    "input_dim": 341,
                    "output_dim": 101,
                    "hidden_layers": [],
                    "activation": "relu",
                    "dropout": 0.5,
                    "residual": False,
                    "attention": False,
                    "feature_interaction": False,
                    "normalisation_in_network": None,
                    "kan_config": None,
                    "tabnet_config": None,
                    "classical_ml_config": None,
                    "pseudo_inverse_config": None
                }
            },
            "training": {
                "loss": {
                    "type": "cross_entropy",
                    "class_weights": "inverse_frequency",
                    "label_smoothing": 0.0
                },
                "optimizer": {
                    "type": "adamw",
                    "lr": 1e-5,
                    "weight_decay": 1e-5,
                    "betas": [0.9, 0.999]
                },
                "scheduler": {
                    "type": "cosine",
                    "params": {}
                },
                "batch_size": 128,
                "n_epochs": 30,
                "early_stopping": self.SHORT_TERM_EARLY_STOPPING.copy()
            },
            "hardware": {
                "gpu": self._get_gpu_name(),
                "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "train_time_hours": None,
                "inference_time_s_per_1e6_voxels": None
            },
            "results": {
                "evaluated_split": "test",
                "global_metrics": {
                    "gc": None,                     # Gross Classification (Top-1 Accuracy)
                    "top_3_accuracy": None,         # Top-3 Accuracy
                    "balanced_accuracy": None,
                    "macro_f1": None,
                    "weighted_f1": None,
                    "kappa": None,                  # Cohen's Kappa
                    "nll": None,
                    "brier_score": None,
                    "ece": None
                },
                "per_class_metrics_path": None,
                "per_subject_metrics_path": None,
                "confusion_matrix_path": None,
                "curves": {
                    "reliability_diagram_path": None,
                    "risk_coverage_curve_path": None
                },
                "logs": {
                    "train_curve_path": None,
                    "val_curve_path": None
                },
                "notes": ""
            }
        }

    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"

    def _get_gpu_name(self) -> str:
        """Get GPU name if available."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"

    def set_from_config(self, config: Dict[str, Any]) -> None:
        """
        Set experiment parameters from the training config dictionary.

        Args:
            config: Training configuration dictionary
        """
        # Task configuration
        self.data["seed"] = config.get("random_seed", 666)

        # Determine label space
        num_class = config.get("num_class", 102)
        if num_class == 101:
            self.data["task"]["label_space"] = "alex-101"
        elif num_class == 102:
            self.data["task"]["label_space"] = "alex-102"
        elif num_class == 100:
            self.data["task"]["label_space"] = "alex-100"

        # Split configuration (support both old and new config formats)
        # New format: test_subject (string)
        if "test_subject" in config:
            self.data["task"]["split"]["test_subjects"] = [config["test_subject"]]
        # Old format: test_prob_idx (list)
        elif "test_prob_idx" in config:
            self.data["task"]["split"]["test_subjects"] = config.get("test_prob_idx", [])

        if "val_prob_idx" in config:
            self.data["task"]["split"]["val_subjects"] = config.get("val_prob_idx", [])

        # Preprocessing - support both old and new config formats
        # New format: use_zscore (bool)
        if config.get("use_zscore", True):
            self.data["preprocessing"]["normalisation"]["type"] = "per_patient_zscore"
            self.data["preprocessing"]["normalisation"]["params"] = {
                "epsilon": 1e-8,
                "fit_scope": "per_subject"
            }
        # Old format: standardization_method (string)
        elif config.get("standardization_method") == "global":
            self.data["preprocessing"]["normalisation"]["type"] = "global_standard_scaler"
            self.data["preprocessing"]["normalisation"]["params"] = {
                "epsilon": 1e-8,
                "fit_scope": "train_global"
            }

        # Background handling (default: include all classes)
        self.data["preprocessing"]["background_handling"] = "include_all_classes"

        # Model configuration
        model_type = config.get("model_type", "deep_mlp")
        hidden_units = config.get("hidden_units", [2048, 2048, 2048, 2048, 2048, 2048])

        self.data["model"]["details"]["input_dim"] = config.get("feature_dim", 341)
        self.data["model"]["details"]["output_dim"] = num_class
        self.data["model"]["details"]["hidden_layers"] = hidden_units
        self.data["model"]["details"]["activation"] = config.get("activation", "gelu")
        self.data["model"]["details"]["dropout"] = config.get("dropout_rate", 0.25)

        # Set model-specific flags
        if model_type == "residual_mlp":
            self.data["model"]["details"]["residual"] = True

        # Training configuration
        self.data["training"]["optimizer"]["type"] = config.get("optimizer", "adamw")
        self.data["training"]["optimizer"]["lr"] = config.get("lr", 1e-5)
        self.data["training"]["optimizer"]["weight_decay"] = config.get("weight_decay", 1e-5)

        # Scheduler
        if config.get("use_lr_scheduler", True):
            scheduler_type = config.get("lr_scheduler_type", "cosine")
            self.data["training"]["scheduler"]["type"] = scheduler_type
        else:
            self.data["training"]["scheduler"]["type"] = "none"

        self.data["training"]["batch_size"] = config.get("batch_size", 128)
        self.data["training"]["n_epochs"] = config.get("epochs", 30)

        # Select early stopping template based on n_epochs
        n_epochs = config.get("epochs", 30)
        if n_epochs >= 80:
            self.data["training"]["early_stopping"] = self.LONG_TERM_EARLY_STOPPING.copy()
        else:
            self.data["training"]["early_stopping"] = self.SHORT_TERM_EARLY_STOPPING.copy()

    def set_voxel_counts(
        self,
        train_count: int,
        val_count: int,
        test_count: int
    ) -> None:
        """Set the number of voxels in each split."""
        self.data["task"]["n_voxels"]["train"] = train_count
        self.data["task"]["n_voxels"]["val"] = val_count
        self.data["task"]["n_voxels"]["test"] = test_count

    def set_train_subjects(self, subjects: List[int]) -> None:
        """Set the training subject IDs."""
        self.data["task"]["split"]["train_subjects"] = subjects

    def set_model_param_count(self, model: torch.nn.Module) -> None:
        """Calculate and set model parameter count."""
        total_params = sum(p.numel() for p in model.parameters())
        self.data["model"]["param_count_m"] = round(total_params / 1e6, 2)

    def set_training_time(self, seconds: float) -> None:
        """Set training time in hours."""
        self.data["hardware"]["train_time_hours"] = round(seconds / 3600, 2)

    def log_results(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        split: str = "test",
        loss_val: Optional[float] = None,
        additional_notes: str = ""
    ) -> None:
        """
        Calculate and log evaluation results.

        Args:
            y_true: Ground truth labels (N,)
            y_pred_proba: Predicted probabilities (N, n_classes)
            split: Which split was evaluated ("train", "val", "test")
            loss_val: Optional loss value
            additional_notes: Any additional notes to add
        """
        self.data["results"]["evaluated_split"] = split

        # Calculate metrics using MetricCalculator if available
        if MetricCalculator is not None:
            n_classes = self.data["model"]["details"]["output_dim"]
            metrics = MetricCalculator.compute_all(y_true, y_pred_proba, loss_val, n_classes)

            # Map metrics to schema (using schema-compliant field names)
            self.data["results"]["global_metrics"] = {
                "gc": metrics.get("gross_accuracy"),           # Gross Classification
                "top_3_accuracy": metrics.get("top3_accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "kappa": metrics.get("cohen_kappa"),           # Cohen's Kappa
                "nll": metrics.get("nll"),
                "brier_score": metrics.get("brier_score"),
                "ece": metrics.get("ece")
            }
        else:
            # Fallback: compute basic metrics
            from sklearn.metrics import (
                accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score
            )
            y_pred = np.argmax(y_pred_proba, axis=1)

            self.data["results"]["global_metrics"]["gc"] = float(accuracy_score(y_true, y_pred))
            self.data["results"]["global_metrics"]["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
            self.data["results"]["global_metrics"]["macro_f1"] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            self.data["results"]["global_metrics"]["weighted_f1"] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            self.data["results"]["global_metrics"]["kappa"] = float(cohen_kappa_score(y_true, y_pred))

        # Add notes
        notes = f"Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if additional_notes:
            notes += f"\n{additional_notes}"
        self.data["results"]["notes"] = notes

    def log_results_from_eval_dict(
        self,
        eval_results: Dict[str, Any],
        split: str = "test"
    ) -> None:
        """
        Log results from the evaluate_model() return dictionary.

        Args:
            eval_results: Dictionary returned by evaluate_model()
            split: Which split was evaluated
        """
        self.data["results"]["evaluated_split"] = split

        # Map from evaluate_model output to schema (using schema-compliant field names)
        self.data["results"]["global_metrics"]["gc"] = eval_results.get("accuracy")
        self.data["results"]["global_metrics"]["balanced_accuracy"] = eval_results.get("balanced_accuracy")
        self.data["results"]["global_metrics"]["macro_f1"] = eval_results.get("f1_macro")
        self.data["results"]["global_metrics"]["weighted_f1"] = eval_results.get("f1_weighted")
        self.data["results"]["global_metrics"]["kappa"] = eval_results.get("kappa")

        # If probabilities are available, compute additional metrics
        if "targets" in eval_results and "probabilities" in eval_results:
            y_true = eval_results["targets"]
            y_pred_proba = eval_results["probabilities"]

            if len(y_pred_proba) > 0 and MetricCalculator is not None:
                n_classes = self.data["model"]["details"]["output_dim"]
                metrics = MetricCalculator.compute_all(y_true, y_pred_proba, n_classes=n_classes)

                self.data["results"]["global_metrics"]["top_3_accuracy"] = metrics.get("top3_accuracy")
                self.data["results"]["global_metrics"]["nll"] = metrics.get("nll")
                self.data["results"]["global_metrics"]["ece"] = metrics.get("ece")
                self.data["results"]["global_metrics"]["brier_score"] = metrics.get("brier_score")
            elif MetricCalculator is None:
                print("WARNING: MetricCalculator not available, top_3_accuracy/nll/ece/brier_score will be None")

    def set_result_paths(
        self,
        save_dir: str,
        experiment_name: str
    ) -> None:
        """
        Set the paths for result files based on save directory.

        Args:
            save_dir: Base directory where results are saved
            experiment_name: Name of the experiment (used in filenames)
        """
        self.data["results"]["per_class_metrics_path"] = f"{save_dir}/test_class_metrics.csv"
        self.data["results"]["confusion_matrix_path"] = f"{save_dir}/test_confusion_matrix.png"
        self.data["results"]["logs"]["train_curve_path"] = f"{save_dir}/{experiment_name}_metrics.csv"
        self.data["results"]["logs"]["val_curve_path"] = f"{save_dir}/{experiment_name}_metrics.csv"

    def add_bayesopt_info(self, study_name: str, best_trial_number: int) -> None:
        """
        Add BayesOpt selection information to notes.

        Args:
            study_name: Name of the Optuna study
            best_trial_number: Trial number of the best configuration
        """
        bayesopt_note = f"Selected by: {study_name}, best_trial: {best_trial_number}"
        if self.data["results"]["notes"]:
            self.data["results"]["notes"] += f"\n{bayesopt_note}"
        else:
            self.data["results"]["notes"] = bayesopt_note

    def validate(self) -> tuple:
        """
        Validate the experiment JSON against the schema.

        Returns:
            (is_valid, list_of_errors)
        """
        if SchemaValidator is not None:
            return SchemaValidator.validate(self.data)
        else:
            # Basic validation
            errors = []
            required_top = ["experiment_id", "method_name", "family", "subfamily", "results"]
            for field in required_top:
                if field not in self.data or self.data[field] is None:
                    errors.append(f"Missing required field: {field}")
            return len(errors) == 0, errors

    def get_missing_fields(self) -> List[str]:
        """
        Get list of fields that are still None or empty.

        Returns:
            List of field paths that need to be filled
        """
        missing = []

        def check_dict(d, prefix=""):
            for key, value in d.items():
                path = f"{prefix}.{key}" if prefix else key
                if value is None:
                    missing.append(path)
                elif isinstance(value, dict):
                    check_dict(value, path)
                elif isinstance(value, list) and len(value) == 0:
                    if key not in ["train_subjects", "val_subjects", "test_subjects"]:
                        missing.append(f"{path} (empty list)")

        check_dict(self.data)
        return missing

    def save(self, output_dir: str, validate: bool = True) -> str:
        """
        Save the experiment JSON to file.

        Args:
            output_dir: Directory to save the JSON file
            validate: Whether to run schema validation before saving

        Returns:
            Path to the saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"{self.data['experiment_id']}.json"

        # Validation
        if validate:
            is_valid, errors = self.validate()
            if not is_valid:
                print(f"\n Schema validation warnings for {self.data['experiment_id']}:")
                for err in errors:
                    print(f"  - {err}")
                print("  (File will still be saved)\n")
            else:
                print(f" Schema validation passed.")

        # Check for missing/TODO fields
        missing = self.get_missing_fields()
        if missing:
            print(f"\n Fields still requiring values:")
            for field in missing[:10]:  # Show first 10
                print(f"  - {field}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

        # Save
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        print(f" Experiment JSON saved to: {filepath}")
        return str(filepath)

    def to_dict(self) -> Dict[str, Any]:
        """Return the experiment data as a dictionary."""
        return self.data.copy()


def create_experiment_logger_from_config(
    config: Dict[str, Any],
    experiment_id: Optional[str] = None,
    method_name: Optional[str] = None,
    method_key: Optional[str] = None,
    subfamily: Optional[str] = None
) -> ExperimentJSONLogger:
    """
    Factory function to create an ExperimentJSONLogger from a config dict.

    Args:
        config: Training configuration dictionary
        experiment_id: Override experiment ID (default: auto-generated)
        method_name: Override method name
        method_key: Override method key
        subfamily: Override subfamily

    Returns:
        Configured ExperimentJSONLogger instance
    """
    # Auto-generate IDs if not provided
    model_type = config.get("model_type", "base_mlp")
    hidden_units = config.get("hidden_units", [4096, 4096, 4096, 4096])
    std_method = config.get("standardization_method", "patientwise")

    # Build subfamily from architecture
    if subfamily is None:
        depth = len(hidden_units)
        width = hidden_units[0] if hidden_units else 4096
        if model_type == "deep_mlp":
            subfamily = f"deepmlp_{depth}x{width}"
        elif model_type == "residual_mlp":
            subfamily = f"resmlp_{depth}x{width}"
        else:
            subfamily = f"baseline_{depth}x{width}"

    if experiment_id is None:
        experiment_id = f"alex_{subfamily}_{std_method}_v1"

    if method_name is None:
        method_name = f"{model_type.replace('_', ' ').title()} {hidden_units} ({std_method})"

    if method_key is None:
        method_key = f"{subfamily}_{std_method}"

    # Create logger
    logger = ExperimentJSONLogger(
        experiment_id=experiment_id,
        method_name=method_name,
        method_key=method_key,
        family="mlp",
        subfamily=subfamily,
        seed=config.get("random_seed", 666)
    )

    # Configure from config
    logger.set_from_config(config)

    return logger


# Example usage and integration guide
if __name__ == "__main__":
    print("=" * 60)
    print("ExperimentJSONLogger - Example Usage")
    print("=" * 60)

    # Example config (similar to what config.py provides)
    example_config = {
        "random_seed": 666,
        "model_type": "deep_mlp",
        "hidden_units": [2048, 2048, 2048, 2048, 2048, 2048],
        "feature_dim": 341,
        "num_class": 101,
        "activation": "gelu",
        "dropout_rate": 0.25,
        "optimizer": "adamw",
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs": 30,
        "use_lr_scheduler": True,
        "lr_scheduler_type": "cosine",
        "standardization_method": "patientwise",
        "filter_background": False,
        "include_background_in_classes": True,
        "val_prob_idx": [20],
        "test_prob_idx": [38]
    }

    # Create logger
    logger = create_experiment_logger_from_config(
        example_config,
        experiment_id="alex_deepmlp_6x2048_patientwise_v1",
        method_name="Deep MLP 6x2048 (BayesOpt best, GELU)",
        method_key="deepmlp_6x2048_patientwise"
    )

    # Set voxel counts (would come from dataset_dict in real usage)
    logger.set_voxel_counts(
        train_count=5000000,  # Example
        val_count=150000,
        test_count=150000
    )

    # Print what would be saved
    print("\nGenerated Experiment JSON structure:")
    print(json.dumps(logger.data, indent=2, ensure_ascii=False)[:2000] + "...")

    # Check missing fields
    print("\n Fields requiring attention:")
    for field in logger.get_missing_fields()[:15]:
        print(f"  - {field}")
