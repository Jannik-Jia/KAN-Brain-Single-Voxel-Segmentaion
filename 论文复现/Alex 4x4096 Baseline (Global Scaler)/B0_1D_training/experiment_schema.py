#!/usr/bin/env python3
"""
统一的实验记录 Schema 定义
定义了所有实验必须包含的字段和可选字段
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field


# ==================== Schema 定义 ====================

# 所有模型通用的必需字段
REQUIRED_FIELDS = {
    # Meta 信息
    "experiment_id": str,
    "method_name": str,
    "method_key": str,
    "family": str,
    "subfamily": str,
    "code_version": str,
    "seed": int,

    # Task/Data
    "task": {
        "dataset": str,
        "label_space": str,
        "grid": str,
        "split": {
            "scheme": str,
            "split_id": str,
            "train_subjects": list,
            "val_subjects": list,
            "test_subjects": list,
        },
        "n_voxels": {
            "train": int,
            "val": int,
            "test": int,
        }
    },

    # Preprocessing
    "preprocessing": {
        "feature_version": str,
        "normalisation": {
            "type": str,
            "params": dict,
        },
        "background_handling": str,
        "class_weighting": {
            "scheme": str,
            "computed_on": (str, type(None)),
            "weights_file": (str, type(None)),
        }
    },

    # Model
    "model": {
        "family": str,
        "param_count_m": (float, int),
        "details": dict,
    },

    # Training
    "training": {
        "loss": {
            "type": str,
            "class_weights": str,
            "label_smoothing": (float, int),
        },
        "optimizer": {
            "type": str,
            "lr": float,
            "weight_decay": float,
            "betas": list,
        },
        "scheduler": {
            "type": str,
            "params": dict,
        },
        "batch_size": int,
        "n_epochs": int,
        "early_stopping": {
            "enabled": bool,
            "monitor": (str, type(None)),
            "mode": (str, type(None)),
            "min_epochs": (int, type(None)),
            "patience": (int, type(None)),
            "delta": (float, type(None)),
        },
    },

    # Hardware (放在 training 下或独立)
    "hardware": {
        "gpu": str,
        "num_gpus": int,
        "train_time_hours": (float, type(None)),
        "inference_time_s_per_1e6_voxels": (float, type(None)),
    },

    # Results
    "results": {
        "evaluated_split": str,
        "global_metrics": {
            # 基础准确率指标
            "gross_accuracy": (float, type(None)),
            "top1_accuracy": (float, type(None)),
            "top3_accuracy": (float, type(None)),
            "top5_accuracy": (float, type(None)),
            "balanced_accuracy": (float, type(None)),

            # F1 和分割指标
            "macro_f1": (float, type(None)),
            "weighted_f1": (float, type(None)),
            "macro_soft_dice": (float, type(None)),

            # 一致性和校准指标
            "cohen_kappa": (float, type(None)),
            "kappa": (float, type(None)),  # 保留别名兼容性
            "nll": (float, type(None)),
            "ece": (float, type(None)),
            "brier_score": (float, type(None)),

            # 风险覆盖指标
            "risk_at_95_coverage": (float, type(None)),
            "actual_coverage_95": (float, type(None)),

            # 其他指标
            "gc": (float, type(None)),

            # 保留的额外指标（向后兼容）
            "top_3_accuracy": (float, type(None)),  # 别名，映射到 top3_accuracy
        },
        "per_class_metrics_path": (str, type(None)),
        "per_subject_metrics_path": (str, type(None)),
        "confusion_matrix_path": (str, type(None)),
        "curves": {
            "reliability_diagram_path": (str, type(None)),
            "risk_coverage_curve_path": (str, type(None)),
        },
        "logs": {
            "train_curve_path": (str, type(None)),
            "val_curve_path": (str, type(None)),
        },
        "notes": (str, type(None)),
    }
}


# 按 family 定义 model.details 的必需字段
FAMILY_SPECIFIC_FIELDS = {
    "mlp": {
        "input_dim": int,
        "output_dim": int,
        "hidden_layers": list,
        "activation": str,
        "dropout": (float, int),
        "residual": bool,
        "attention": bool,
        "feature_interaction": bool,
        "normalisation_in_network": (str, type(None)),
        "kan_config": type(None),
        "tabnet_config": type(None),
        "classical_ml_config": type(None),
        "pseudo_inverse_config": type(None),
    },

    "kan": {
        "input_dim": int,
        "output_dim": int,
        "hidden_layers": list,
        "kan_config": {
            "grid_size": int,
            "spline_order": int,
            "base_activation": str,
        },
        "dropout": (float, int),
        "normalisation_in_network": (str, type(None)),
        "tabnet_config": type(None),
        "classical_ml_config": type(None),
        "pseudo_inverse_config": type(None),
    },

    "tabnet": {
        "input_dim": int,
        "output_dim": int,
        "tabnet_config": {
            "n_d": int,
            "n_a": int,
            "n_steps": int,
            "gamma": float,
            "n_independent": int,
            "n_shared": int,
            "momentum": float,
        },
        "dropout": (float, int),
        "normalisation_in_network": (str, type(None)),
        "kan_config": type(None),
        "classical_ml_config": type(None),
        "pseudo_inverse_config": type(None),
    },

    "classical_ml": {
        "input_dim": int,
        "output_dim": int,
        "classical_ml_config": {
            "algorithm": str,  # svm, rf, xgboost, etc.
            "params": dict,
        },
        "kan_config": type(None),
        "tabnet_config": type(None),
        "pseudo_inverse_config": type(None),
    },

    "pseudo_inverse": {
        "input_dim": int,
        "output_dim": int,
        "pseudo_inverse_config": {
            "regularization": (float, type(None)),
            "method": str,  # svd, lstsq, etc.
        },
        "kan_config": type(None),
        "tabnet_config": type(None),
        "classical_ml_config": type(None),
    },
}


# ==================== 辅助函数 ====================

def get_nested_value(data: Dict, path: str) -> Any:
    """
    从嵌套字典中获取值
    path: "task.split.scheme" -> data["task"]["split"]["scheme"]
    """
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def check_type(value: Any, expected_type: Union[type, tuple]) -> bool:
    """
    检查值的类型是否匹配
    expected_type 可以是单个类型或类型元组
    """
    if isinstance(expected_type, tuple):
        return isinstance(value, expected_type)
    return isinstance(value, expected_type)


def validate_nested_dict(data: Dict, schema: Dict, path: str = "") -> List[str]:
    """
    递归验证嵌套字典是否符合 schema

    Args:
        data: 要验证的数据
        schema: schema 定义
        path: 当前路径（用于错误消息）

    Returns:
        错误消息列表
    """
    errors = []

    for key, expected in schema.items():
        current_path = f"{path}.{key}" if path else key

        # 检查字段是否存在
        if key not in data:
            errors.append(f"缺少必需字段: {current_path}")
            continue

        value = data[key]

        # 如果 expected 是字典，递归检查
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                errors.append(f"字段类型错误: {current_path} 应该是 dict，实际是 {type(value).__name__}")
            else:
                errors.extend(validate_nested_dict(value, expected, current_path))

        # 否则检查类型
        else:
            if not check_type(value, expected):
                expected_types = expected if isinstance(expected, tuple) else (expected,)
                expected_names = ", ".join(t.__name__ if t is not type(None) else "None"
                                          for t in expected_types)
                errors.append(
                    f"字段类型错误: {current_path} 应该是 [{expected_names}]，"
                    f"实际是 {type(value).__name__}"
                )

    return errors


def validate_experiment_json(experiment: Dict, verbose: bool = True) -> tuple[bool, List[str]]:
    """
    验证实验记录 JSON 是否符合标准 schema

    Args:
        experiment: 实验记录字典
        verbose: 是否打印详细信息

    Returns:
        (is_valid, errors): 是否有效和错误列表
    """
    errors = []

    # 1. 验证通用字段
    if verbose:
        print("=" * 60)
        print("验证实验记录 Schema")
        print("=" * 60)
        print(f"Experiment ID: {experiment.get('experiment_id', 'N/A')}")
        print(f"Method: {experiment.get('method_name', 'N/A')}")
        print(f"Family: {experiment.get('family', 'N/A')}")
        print("-" * 60)

    errors.extend(validate_nested_dict(experiment, REQUIRED_FIELDS))

    # 2. 验证 family 特定字段
    family = experiment.get("family")
    if family and family in FAMILY_SPECIFIC_FIELDS:
        model_details = experiment.get("model", {}).get("details", {})
        family_schema = FAMILY_SPECIFIC_FIELDS[family]

        family_errors = validate_nested_dict(model_details, family_schema, "model.details")
        errors.extend(family_errors)
    elif family and family not in FAMILY_SPECIFIC_FIELDS:
        errors.append(f"未知的 family: {family}，请在 FAMILY_SPECIFIC_FIELDS 中添加定义")

    # 3. 打印结果
    if verbose:
        if errors:
            print(f"\n❌ 发现 {len(errors)} 个问题:\n")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        else:
            print("✅ Schema 验证通过！所有必需字段均存在且类型正确。")
        print("=" * 60)

    return len(errors) == 0, errors


def get_schema_summary() -> str:
    """
    返回 schema 的人类可读摘要
    """
    lines = [
        "=" * 60,
        "实验记录标准 Schema 摘要",
        "=" * 60,
        "",
        "所有实验必需的顶层字段:",
        "  - experiment_id, method_name, method_key",
        "  - family, subfamily",
        "  - code_version, seed",
        "",
        "必需的嵌套部分:",
        "  - task (dataset, label_space, grid, split, n_voxels)",
        "  - preprocessing (feature_version, normalisation, background_handling, class_weighting)",
        "  - model (family, param_count_m, details)",
        "  - training (loss, optimizer, scheduler, batch_size, n_epochs, early_stopping)",
        "  - hardware (gpu, num_gpus, train_time_hours, inference_time_s_per_1e6_voxels)",
        "  - results (evaluated_split, global_metrics, paths, curves, logs)",
        "",
        "支持的 family 类型:",
    ]

    for family in FAMILY_SPECIFIC_FIELDS.keys():
        lines.append(f"  - {family}")

    lines.extend([
        "",
        "每个 family 在 model.details 中有特定的必需字段。",
        "详细信息请参考 FAMILY_SPECIFIC_FIELDS 定义。",
        "=" * 60,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # 打印 schema 摘要
    print(get_schema_summary())

    # 示例：验证一个最小的实验记录
    print("\n\n测试示例验证:")
    test_experiment = {
        "experiment_id": "test_v1",
        "method_name": "Test Method",
        "method_key": "test",
        "family": "mlp",
        "subfamily": "test",
        "code_version": "git-abc123",
        "seed": 42,
        "task": {
            "dataset": "alex",
            "label_space": "alex-102",
            "grid": "mprage",
            "split": {
                "scheme": "subject-wise",
                "split_id": "fold1",
                "train_subjects": [1, 2, 3],
                "val_subjects": [],
                "test_subjects": [4],
            },
            "n_voxels": {
                "train": 1000000,
                "val": 0,
                "test": 100000,
            }
        },
        "preprocessing": {
            "feature_version": "voxel_signature_v1",
            "normalisation": {
                "type": "global_standard_scaler",
                "params": {"epsilon": None, "fit_scope": "train_global"},
            },
            "background_handling": "drop_ignore_index",
            "class_weighting": {
                "scheme": "none",
                "computed_on": None,
                "weights_file": None,
            }
        },
        "model": {
            "family": "mlp",
            "param_count_m": 52.0,
            "details": {
                "input_dim": 341,
                "output_dim": 102,
                "hidden_layers": [4096, 4096, 4096, 4096],
                "activation": "relu",
                "dropout": 0.5,
                "residual": False,
                "attention": False,
                "feature_interaction": False,
                "normalisation_in_network": None,
                "kan_config": None,
                "tabnet_config": None,
                "classical_ml_config": None,
                "pseudo_inverse_config": None,
            }
        },
        "training": {
            "loss": {
                "type": "cross_entropy",
                "class_weights": "none",
                "label_smoothing": 0.0,
            },
            "optimizer": {
                "type": "adam",
                "lr": 1e-5,
                "weight_decay": 1e-5,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "type": "none",
                "params": {},
            },
            "batch_size": 128,
            "n_epochs": 25,
            "early_stopping": {
                "enabled": False,
                "monitor": None,
                "mode": None,
                "min_epochs": None,
                "patience": None,
                "delta": None,
            },
        },
        "hardware": {
            "gpu": "CPU",
            "num_gpus": 0,
            "train_time_hours": None,
            "inference_time_s_per_1e6_voxels": None,
        },
        "results": {
            "evaluated_split": "test",
            "global_metrics": {
                # 基础准确率指标
                "gross_accuracy": 0.70,
                "top1_accuracy": 0.70,
                "top3_accuracy": 0.89,
                "top5_accuracy": 0.95,
                "balanced_accuracy": 0.68,

                # F1 和分割指标
                "macro_f1": 0.75,
                "weighted_f1": 0.78,
                "macro_soft_dice": 0.72,

                # 一致性和校准指标
                "cohen_kappa": 0.65,
                "kappa": 0.65,
                "nll": 0.85,
                "ece": 0.03,
                "brier_score": 0.12,

                # 风险覆盖指标
                "risk_at_95_coverage": 0.08,
                "actual_coverage_95": 0.94,

                # 其他指标
                "gc": None,

                # 向后兼容
                "top_3_accuracy": 0.89,
            },
            "per_class_metrics_path": None,
            "per_subject_metrics_path": None,
            "confusion_matrix_path": None,
            "curves": {
                "reliability_diagram_path": None,
                "risk_coverage_curve_path": None,
            },
            "logs": {
                "train_curve_path": None,
                "val_curve_path": None,
            },
            "notes": "Test experiment",
        }
    }

    is_valid, errors = validate_experiment_json(test_experiment, verbose=True)
