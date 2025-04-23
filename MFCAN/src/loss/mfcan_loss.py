import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MFCANLoss(nn.Module):
    """
    MFCAN的多任务损失函数，结合主分类损失和辅助分类损失，
    并可选择性地添加注意力正则化和类别平衡
    """
    

    def __init__(self, num_classes=102, aux_weight=0.3, attn_reg_weight=0.01, 
             modal_balance_weight=0.01, class_balance_method='effective_samples', 
             focal_gamma=2.0, cb_beta=0.9999, cb_samples_per_class=None):
        """
        初始化MFCAN损失函数
        
        参数:
            num_classes: 类别数量
            aux_weight: 辅助损失权重
            attn_reg_weight: 注意力正则化权重
            class_balance_method: 类别平衡方法，可选 'inverse', 'effective_samples', 'none'
            focal_gamma: Focal Loss的gamma参数
            cb_beta: 类别平衡中的beta参数
            cb_samples_per_class: 每个类别的样本数
        """
        super(MFCANLoss, self).__init__()
        self.num_classes = num_classes
        self.aux_weight = aux_weight
        self.attn_reg_weight = attn_reg_weight
        self.class_balance_method = class_balance_method
        self.focal_gamma = focal_gamma
        self.cb_beta = cb_beta
        self.cb_samples_per_class = cb_samples_per_class
        self.modal_balance_weight = modal_balance_weight

        # 初始化类别权重为None，将在forward中根据需要计算
        self.class_weights = None
    
    def update_class_counts(self, class_counts):
        """更新类别计数并重新计算权重"""
        self.cb_samples_per_class = class_counts
        self._update_weights()
    
    def _update_weights(self):
        """根据选择的方法计算类别权重"""
        if self.cb_samples_per_class is None or self.class_balance_method == 'none':
            self.class_weights = None
            return
            
        class_counts = np.array(self.cb_samples_per_class)
        
        if self.class_balance_method == 'inverse':
            # 简单的反比例权重
            weights = 1.0 / np.maximum(class_counts, 1)  # 避免除零
            weights = weights / weights.sum() * len(weights)  # 归一化
            
        elif self.class_balance_method == 'effective_samples':
            # "Class-Balanced Loss Based on Effective Number of Samples"中的方法
            effective_num = 1.0 - np.power(self.cb_beta, class_counts)
            weights = (1.0 - self.cb_beta) / np.where(effective_num > 0, effective_num, 1e-8)
            weights = weights / weights.sum() * len(weights)  # 归一化
            
        elif self.class_balance_method == 'sqrt_inverse':
            # 平方根反比例权重，减轻极端不平衡影响
            weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
            weights = weights / weights.sum() * len(weights)  # 归一化
            
        else:
            weights = None
            
        if weights is not None:
            self.class_weights = torch.FloatTensor(weights)
        else:
            self.class_weights = None
    
    def forward(self, outputs, targets, class_counts=None):
        """
        计算损失
        
        参数:
            outputs: 模型输出字典
            targets: 目标类别 [batch_size]
            class_counts: 各类别样本数量，用于类别平衡
            
        返回:
            total_loss: 总损失
            loss_info: 各组件损失的字典
        """
        device = targets.device
        
        # 如果提供了新的类别计数，更新权重
        if class_counts is not None and self.class_balance_method != 'none':
            self.cb_samples_per_class = class_counts
            self._update_weights()
            
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
        
        # 主分类损失
        if 'main_output' in outputs:
            main_loss = self._compute_classification_loss(
                outputs['main_output'], targets, self.class_weights)
        else:
            main_loss = torch.tensor(0.0, device=device)
        
        # 辅助分类损失
        aux_losses = {}
        if 'aux_outputs' in outputs:
            for modal, aux_output in outputs['aux_outputs'].items():
                aux_losses[modal] = self._compute_classification_loss(
                    aux_output, targets, self.class_weights)
            
            # 计算平均辅助损失
            avg_aux_loss = sum(aux_losses.values()) / len(aux_losses)
        else:
            avg_aux_loss = torch.tensor(0.0, device=device)
        
        # 注意力正则化损失
        attn_reg_loss = torch.tensor(0.0, device=device)
        attn_balance_loss = torch.tensor(0.0, device=device)

        if 'attention_weights' in outputs and self.attn_reg_weight > 0:
            attn_weights = outputs['attention_weights']  # [batch_size, num_modalities]
            
            # 熵正则化 - 鼓励更确定的注意力分配
            attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-6), dim=1).mean()
            attn_reg_loss = self.attn_reg_weight * attn_entropy
            
            # 模态平衡正则化 - 鼓励使用所有模态
            if self.modal_balance_weight > 0:
                # 计算批次平均注意力
                mean_attn = torch.mean(attn_weights, dim=0)  # [num_modalities]
                # 计算与均匀分布的KL散度
                uniform_attn = torch.ones_like(mean_attn) / mean_attn.size(0)
                modal_balance_loss = torch.sum(mean_attn * torch.log(mean_attn / uniform_attn + 1e-6))
                attn_balance_loss = self.modal_balance_weight * modal_balance_loss
        
        # 计算总损失
        if 'main_output' in outputs and 'aux_outputs' in outputs:
            total_loss = main_loss + self.aux_weight * avg_aux_loss + attn_reg_loss + attn_balance_loss  
        elif 'aux_outputs' in outputs:
            # 如果只有辅助输出（编码器预训练阶段）
            total_loss = avg_aux_loss
        else:
            # 如果只有主输出
            total_loss = main_loss + attn_reg_loss
        
        # 返回损失信息
        loss_info = {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item() if main_loss.requires_grad else 0.0,
            'avg_aux_loss': avg_aux_loss.item() if isinstance(avg_aux_loss, torch.Tensor) and avg_aux_loss.requires_grad else 0.0,
            'attn_reg_loss': attn_reg_loss.item() if attn_reg_loss.requires_grad else 0.0,
            'aux_losses': {k: v.item() for k, v in aux_losses.items()} if aux_losses else {}
        }
        
        return total_loss, loss_info
    
    def _compute_classification_loss(self, outputs, targets, class_weights=None):
        """
        计算分类损失，支持Focal Loss和类别权重
        
        参数:
            outputs: 分类器输出 [batch_size, num_classes]
            targets: 目标类别 [batch_size]
            class_weights: 类别权重
            
        返回:
            loss: 分类损失
        """
        # 使用交叉熵损失
        if self.focal_gamma <= 0 or self.focal_gamma is None:
            return F.cross_entropy(outputs, targets, weight=class_weights, reduction='mean')
        
        # 使用Focal Loss
        # 首先计算交叉熵
        ce_loss = F.cross_entropy(outputs, targets, weight=class_weights, reduction='none')
        
        # 计算样本权重
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.focal_gamma * ce_loss
        
        # 返回平均损失
        return loss.mean()
    

class ClassBalancedLoss(nn.Module):
    """
    类别平衡损失，根据类别频率调整损失权重
    可以结合交叉熵损失或Focal Loss
    """
    
    def __init__(self, num_classes=102, beta=0.9999, samples_per_class=None, 
                 focal_gamma=0.0, device='cuda'):
        """
        初始化类别平衡损失
        
        参数:
            num_classes: 类别数量
            beta: 控制有效样本数的超参数，接近1表示更强的再平衡
            samples_per_class: 每个类别的样本数量
            focal_gamma: Focal Loss的gamma参数，0表示使用交叉熵
            device: 计算设备
        """
        super(ClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.focal_gamma = focal_gamma
        self.device = device
        self.samples_per_class = samples_per_class
        self.weights = None
        
        # 如果提供了每个类别的样本数，计算类别权重
        if samples_per_class is not None:
            self._update_weights()
    
    def _update_weights(self):
        """根据类别样本数更新权重"""
        if self.samples_per_class is None:
            self.weights = None
            return
        
        # 计算有效样本数
        effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        weights = (1.0 - self.beta) / np.array(effective_num)
        
        # 归一化权重
        weights = weights / np.sum(weights) * len(weights)
        
        self.weights = torch.FloatTensor(weights).to(self.device)
    
    def update_samples_per_class(self, samples_per_class):
        """更新每个类别的样本数量"""
        self.samples_per_class = samples_per_class
        self._update_weights()
    
    def forward(self, outputs, targets):
        """
        计算损失
        
        参数:
            outputs: 分类器输出 [batch_size, num_classes]
            targets: 目标类别 [batch_size]
            
        返回:
            loss: 分类损失
        """
        # 使用交叉熵损失
        if self.focal_gamma <= 0 or self.focal_gamma is None:
            return F.cross_entropy(outputs, targets, weight=self.weights, reduction='mean')
        
        # 使用Focal Loss
        ce_loss = F.cross_entropy(outputs, targets, weight=self.weights, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.focal_gamma * ce_loss
        
        return loss.mean()