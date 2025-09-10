"""
Loss functions for imbalanced brain region classification

Implements advanced loss functions designed to handle severe class imbalance
in MRI brain voxel classification, particularly for improving Macro-F1 and
Macro-AUPRC performance on minority classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Paper: "Focal Loss for Dense Object Detection" - Lin et al. (2017)
    
    Focuses learning on hard examples by down-weighting easy examples.
    """
    
    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: Class weighting factors (num_classes,). None for no weighting.
            gamma: Focusing parameter. Higher gamma = more focus on hard examples.
            reduction: Reduction method ('mean', 'sum', 'none')
            label_smoothing: Label smoothing factor [0, 1)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits (N, C) where C = number of classes
            targets: Ground truth labels (N,) with values in [0, C-1]
        
        Returns:
            Focal loss value
        """
        if self.label_smoothing > 0:
            # With label smoothing, we need custom implementation
            return self._focal_loss_with_smoothing(inputs, targets)
        else:
            # Standard focal loss without smoothing
            return self._standard_focal_loss(inputs, targets)
    
    def _standard_focal_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Standard focal loss implementation"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _focal_loss_with_smoothing(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Focal loss with label smoothing implementation"""
        num_classes = inputs.size(1)
        batch_size = inputs.size(0)
        
        # Get smoothed target distribution
        smooth_targets = self._smooth_labels(targets, num_classes)
        
        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = F.softmax(inputs, dim=1)
        
        # Compute cross entropy with smoothed targets
        ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        
        # For focal weight, use the probability of the true class (not smoothed)
        true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - true_class_probs) ** self.gamma
        
        # Apply class weights (alpha) if specified
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _smooth_labels(self, targets: Tensor, num_classes: int) -> Tensor:
        """Apply label smoothing - Convert targets to smoothed probability distribution"""
        batch_size = targets.size(0)
        device = targets.device
        
        # Create one-hot encoding
        smooth_targets = torch.zeros(batch_size, num_classes, device=device, dtype=torch.float)
        smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        return smooth_targets


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss
    
    Paper: "Class-Balanced Loss Based on Effective Number of Samples" - Cui et al. (2019)
    
    Combines class balancing based on effective number of samples with focal loss.
    Particularly effective for long-tailed distributions.
    """
    
    def __init__(
        self,
        class_counts: Tensor,
        beta: float = 0.9999,
        gamma: float = 1.5,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            class_counts: Number of samples per class (num_classes,)
            beta: Re-weighting parameter. Higher beta = more re-weighting
            gamma: Focal loss focusing parameter
            reduction: Reduction method
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # Compute class-balanced weights
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        self.register_buffer('class_weights', weights / weights.sum() * len(weights))
        
        # Initialize focal loss with computed weights
        self.focal_loss = FocalLoss(
            alpha=self.class_weights,
            gamma=gamma,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Forward pass using focal loss with class-balanced weights"""
        return self.focal_loss(inputs, targets)


class LogitAdjustedCrossEntropy(nn.Module):
    """
    Logit-Adjusted Cross-Entropy Loss
    
    Paper: "Long-tail Learning via Logit Adjustment" - Menon et al. (2020)
    
    Adjusts logits based on prior class frequencies to improve tail performance.
    """
    
    def __init__(
        self,
        class_counts: Tensor,
        tau: float = 1.0,
        label_smoothing: float = 0.05,
        reduction: str = 'mean'
    ):
        """
        Args:
            class_counts: Number of samples per class (num_classes,)
            tau: Temperature parameter for logit adjustment
            label_smoothing: Label smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.tau = tau
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # Compute logit adjustment based on class frequencies
        class_priors = class_counts / class_counts.sum()
        logit_adjustment = tau * torch.log(class_priors + 1e-8)
        self.register_buffer('logit_adjustment', logit_adjustment)
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits (N, C)
            targets: Ground truth labels (N,)
        
        Returns:
            Logit-adjusted cross-entropy loss
        """
        # Adjust logits
        adjusted_logits = inputs + self.logit_adjustment
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            return self._smooth_cross_entropy(adjusted_logits, targets)
        else:
            return F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)
    
    def _smooth_cross_entropy(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy with label smoothing"""
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        
        # One-hot encoding with smoothing
        targets_smooth = torch.zeros_like(inputs).scatter_(
            1, targets.unsqueeze(1), 1.0 - self.label_smoothing
        )
        targets_smooth += self.label_smoothing / num_classes
        
        loss = -(targets_smooth * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss
    
    Re-balances the softmax probabilities based on class frequencies.
    Simple but effective approach for imbalanced classification.
    """
    
    def __init__(
        self,
        class_counts: Tensor,
        reduction: str = 'mean'
    ):
        """
        Args:
            class_counts: Number of samples per class (num_classes,)
            reduction: Reduction method
        """
        super().__init__()
        self.reduction = reduction
        
        # Compute class frequencies
        class_freq = class_counts / class_counts.sum()
        self.register_buffer('class_freq', class_freq)
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits (N, C)
            targets: Ground truth labels (N,)
        
        Returns:
            Balanced softmax loss
        """
        # Compute balanced softmax
        logits_adjusted = inputs - torch.log(self.class_freq + 1e-8)
        
        return F.cross_entropy(logits_adjusted, targets, reduction=self.reduction)


class MixupLoss(nn.Module):
    """
    Mixup Loss for data augmentation
    
    Paper: "mixup: Beyond Empirical Risk Minimization" - Zhang et al. (2017)
    
    Can be combined with any base loss function.
    """
    
    def __init__(self, base_loss: nn.Module):
        """
        Args:
            base_loss: Base loss function to apply mixup to
        """
        super().__init__()
        self.base_loss = base_loss
    
    def forward(
        self,
        inputs: Tensor,
        targets_a: Tensor,
        targets_b: Tensor,
        lam: float
    ) -> Tensor:
        """
        Args:
            inputs: Mixed logits (N, C)
            targets_a: First set of labels (N,)
            targets_b: Second set of labels (N,)
            lam: Mixing parameter
        
        Returns:
            Mixed loss
        """
        loss_a = self.base_loss(inputs, targets_a)
        loss_b = self.base_loss(inputs, targets_b)
        return lam * loss_a + (1 - lam) * loss_b


def create_loss_function(
    loss_type: str,
    class_counts: Optional[Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss function
        class_counts: Class sample counts for weighted losses
        **kwargs: Additional parameters for specific loss functions
    
    Returns:
        Loss function module
    """
    
    if loss_type == 'ce':
        # Standard Cross-Entropy
        return nn.CrossEntropyLoss(reduction=kwargs.get('reduction', 'mean'))
    
    elif loss_type == 'weighted_ce':
        # Weighted Cross-Entropy
        if class_counts is None:
            raise ValueError("class_counts required for weighted_ce")
        
        # Compute inverse frequency weights
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum() * len(weights)
        
        return nn.CrossEntropyLoss(
            weight=weights,
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'focal':
        # Focal Loss
        alpha = None
        if class_counts is not None:
            # Compute alpha weights
            alpha = 1.0 / (class_counts + 1e-8)
            alpha = alpha / alpha.sum() * len(alpha)
        
        return FocalLoss(
            alpha=alpha,
            gamma=kwargs.get('gamma', 2.0),
            reduction=kwargs.get('reduction', 'mean'),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )
    
    elif loss_type == 'cb_focal':
        # Class-Balanced Focal Loss
        if class_counts is None:
            raise ValueError("class_counts required for cb_focal")
        
        return ClassBalancedFocalLoss(
            class_counts=class_counts,
            beta=kwargs.get('beta', 0.9999),
            gamma=kwargs.get('gamma', 1.5),
            reduction=kwargs.get('reduction', 'mean'),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )
    
    elif loss_type == 'logit_adj':
        # Logit-Adjusted Cross-Entropy
        if class_counts is None:
            raise ValueError("class_counts required for logit_adj")
        
        return LogitAdjustedCrossEntropy(
            class_counts=class_counts,
            tau=kwargs.get('tau', 1.0),
            label_smoothing=kwargs.get('label_smoothing', 0.05),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'balanced_softmax':
        # Balanced Softmax Loss
        if class_counts is None:
            raise ValueError("class_counts required for balanced_softmax")
        
        return BalancedSoftmaxLoss(
            class_counts=class_counts,
            reduction=kwargs.get('reduction', 'mean')
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def mixup_data(x: Tensor, y: Tensor, alpha: float = 0.2):
    """
    Generate mixed data for Mixup augmentation
    
    Args:
        x: Input batch (N, C, H, W)
        y: Labels (N,)
        alpha: Beta distribution parameter
    
    Returns:
        Mixed inputs, original labels, shuffled labels, mixing parameter
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


if __name__ == "__main__":
    # Test loss functions
    torch.manual_seed(42)
    
    # Simulate imbalanced class distribution
    num_classes = 102
    batch_size = 128
    
    # Create synthetic class counts (imbalanced)
    class_counts = torch.tensor([1000] * 10 + [100] * 20 + [10] * 72, dtype=torch.float)
    print(f"Class distribution - Min: {class_counts.min()}, Max: {class_counts.max()}")
    
    # Generate synthetic data
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test different loss functions
    loss_functions = {
        'ce': create_loss_function('ce'),
        'weighted_ce': create_loss_function('weighted_ce', class_counts=class_counts),
        'focal': create_loss_function('focal', class_counts=class_counts, gamma=2.0),
        'cb_focal': create_loss_function('cb_focal', class_counts=class_counts, beta=0.9999),
        'logit_adj': create_loss_function('logit_adj', class_counts=class_counts, tau=1.0),
        'balanced_softmax': create_loss_function('balanced_softmax', class_counts=class_counts)
    }
    
    print("\n=== Loss Function Comparison ===")
    for name, loss_fn in loss_functions.items():
        loss_value = loss_fn(logits, labels)
        print(f"{name:>15}: {loss_value:.4f}")
    
    # Test mixup
    print("\n=== Mixup Test ===")
    mixed_x, y_a, y_b, lam = mixup_data(torch.randn(4, 351, 7, 7), labels[:4], alpha=0.2)
    print(f"Original shape: {torch.randn(4, 351, 7, 7).shape}")
    print(f"Mixed shape: {mixed_x.shape}")
    print(f"Lambda: {lam:.3f}")
    
    # Test mixup loss
    base_loss = create_loss_function('cb_focal', class_counts=class_counts)
    mixup_loss = MixupLoss(base_loss)
    mixed_logits = torch.randn(4, num_classes)
    mixed_loss = mixup_loss(mixed_logits, y_a, y_b, lam)
    print(f"Mixup loss: {mixed_loss:.4f}")