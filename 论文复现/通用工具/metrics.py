import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, log_loss
)

class MetricCalculator:
    """通用指标计算器，处理所有数学逻辑"""
    
    @staticmethod
    def compute_all(y_true, y_pred_proba, loss_val=None, n_classes=102):
        """
        一次性计算所有标准指标
        Args:
            y_true: (N,) 真实标签
            y_pred_proba: (N, C) 预测概率 (Softmax后)
            loss_val: (float) 外部传入的 loss 值
        Returns:
            dict: 包含所有 schema 要求指标的字典
        """
        # 确保格式正确
        if isinstance(y_pred_proba, list): y_pred_proba = np.array(y_pred_proba)
        if isinstance(y_true, list): y_true = np.array(y_true)
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        confidences = np.max(y_pred_proba, axis=1)

        # 1. 基础准确率
        top1 = accuracy_score(y_true, y_pred)
        balanced = balanced_accuracy_score(y_true, y_pred)
        
        # Top-K
        top3 = MetricCalculator.top_k_accuracy(y_true, y_pred_proba, k=3)
        top5 = MetricCalculator.top_k_accuracy(y_true, y_pred_proba, k=5)

        # 2. F1 & Dice
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        soft_dice = MetricCalculator.soft_dice(y_true, y_pred_proba, n_classes)

        # 3. 校准与一致性
        kappa = cohen_kappa_score(y_true, y_pred)
        nll = log_loss(y_true, y_pred_proba, labels=list(range(n_classes)))
        ece = MetricCalculator.expected_calibration_error(y_true, y_pred_proba)
        brier = MetricCalculator.brier_score(y_true, y_pred_proba, n_classes)

        # 4. 风险覆盖
        risk_95 = MetricCalculator.risk_at_coverage(y_true, y_pred, confidences, target_coverage=0.95)
        cov_05 = MetricCalculator.coverage_at_risk(y_true, y_pred, confidences, target_risk=0.05)

        return {
            "gross_accuracy": float(top1),
            "top1_accuracy": float(top1),
            "top3_accuracy": float(top3),
            "top5_accuracy": float(top5),
            "balanced_accuracy": float(balanced),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_soft_dice": float(soft_dice),
            "cohen_kappa": float(kappa),
            "nll": float(nll),
            "ece": float(ece),
            "brier_score": float(brier),
            "risk_at_95_coverage": float(risk_95),
            "actual_coverage_95": float(cov_05),
            "loss": float(loss_val) if loss_val is not None else None
        }

    @staticmethod
    def top_k_accuracy(y_true, y_proba, k=3):
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        return float(np.any(top_k_preds == y_true[:, None], axis=1).mean())

    @staticmethod
    def soft_dice(y_true, y_proba, n_classes, smooth=1e-6):
        y_true_one_hot = np.eye(n_classes)[y_true]
        intersection = np.sum(y_true_one_hot * y_proba, axis=0)
        dice = (2 * intersection + smooth) / (np.sum(y_true_one_hot, axis=0) + np.sum(y_proba, axis=0) + smooth)
        return float(dice.mean())

    @staticmethod
    def expected_calibration_error(y_true, y_proba, n_bins=10):
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if mask.sum() > 0:
                ece += (mask.sum() / len(y_true)) * abs(accuracies[mask].mean() - confidences[mask].mean())
        return float(ece)

    @staticmethod
    def brier_score(y_true, y_proba, n_classes):
        y_true_one_hot = np.eye(n_classes)[y_true]
        return float(np.mean(np.sum((y_proba - y_true_one_hot) ** 2, axis=1)))

    @staticmethod
    def risk_at_coverage(y_true, y_pred, confidences, target_coverage=0.95):
        # 风险 = 1 - Accuracy
        n = len(y_true)
        k = int(n * target_coverage)
        if k == 0: return 0.0
        indices = np.argsort(-confidences)[:k] # 取置信度最高的k个
        acc = (y_pred[indices] == y_true[indices]).mean()
        return 1.0 - acc

    @staticmethod
    def coverage_at_risk(y_true, y_pred, confidences, target_risk=0.05):
        # 找到满足 Error <= target_risk 的最大覆盖率
        indices = np.argsort(-confidences)
        sorted_correct = (y_pred[indices] == y_true[indices])
        
        # 计算累积错误率
        cumulative_correct = np.cumsum(sorted_correct)
        ns = np.arange(1, len(y_true) + 1)
        cumulative_risk = 1.0 - (cumulative_correct / ns)
        
        valid_indices = np.where(cumulative_risk <= target_risk)[0]
        if len(valid_indices) == 0: return 0.0
        return float((valid_indices[-1] + 1) / len(y_true))