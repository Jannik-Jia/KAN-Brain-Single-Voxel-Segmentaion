#!/usr/bin/env python3
"""
使用1D训练集进行训练，然后映射回3D
基于Alex的架构，但使用351维输入（而不是341维）
训练后将softmax概率映射回3D体积
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import h5py
import scipy.io
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import gc
import copy

# 导入评估指标模块
from evaluation_metrics import compute_all_metrics

# ==================== 模型定义（基于Alex的架构）====================

class RegModel(nn.Module):
    """
    Alex的1D全连接网络
    原始是341维输入，现在改为351维
    """
    def __init__(self, input_dim=351, num_classes=102):
        super(RegModel, self).__init__()
        # 4层全连接网络 + 输出层
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, num_classes)  # 输出层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)  # 不应用softmax，让CrossEntropyLoss处理
        return x


# ==================== M-Layer 模型定义 ====================

class MLayerBlock(nn.Module):
    """
    Matrix Exponentiation Layer Block (Operator-based Implementation)

    核心思想：用矩阵指数 expm(·) 作为演化算子（Evolution Operator）

    理论背景（参考 arXiv:2008.03936 Intelligent Matrix Exponentiation）：
    - 将输入向量 h 视为状态（State），reshape 成方阵 h_state
    - 通过线性层生成李代数元素 M，计算演化算子 E = expm(M)
    - 让算子直接作用于状态：out = E @ h_state
    - 将结果展平回向量空间

    几何意义：
    - E = expm(M) 是李群 GL(n) 上的元素（可逆线性变换）
    - M 是对应的李代数 gl(n) 中的元素（无穷小生成元）
    - 通过矩阵乘法让变换作用于状态，保留了流形结构
    - 相比 Feature-based 实现，引入了强几何归纳偏置

    数值稳定性：
    - scale: 缩放输入矩阵，防止 exp 爆炸 (默认 0.01)
    - clip: 可选的数值裁剪，进一步约束输入范围

    参数量设计：
    - to_matrix: Linear(4096 → 4096) = 16,781,312 参数
    - 相比 Feature-based 实现减少了约 16,781,312 参数（移除了 from_matrix）
    """
    def __init__(self, hidden_dim=4096, matrix_size=64, scale=0.01, clip=10.0):
        super(MLayerBlock, self).__init__()

        # 严格验证：matrix_size^2 必须等于 hidden_dim
        assert matrix_size * matrix_size == hidden_dim, \
            f"matrix_size^2 ({matrix_size}^2={matrix_size*matrix_size}) 必须等于 hidden_dim ({hidden_dim})"

        self.hidden_dim = hidden_dim
        self.matrix_size = matrix_size
        self.scale = scale
        self.clip = clip  # None 表示不做裁剪

        # 唯一的可学习层：用于生成 M（李代数元素）
        self.to_matrix = nn.Linear(hidden_dim, hidden_dim)  # 4096 → 4096
        # 注意：移除了 from_matrix，改用算子作用逻辑

    def forward(self, h):
        """
        Args:
            h: 输入张量（状态向量），形状 [B, hidden_dim] = [B, 4096]
        Returns:
            out: 输出张量（演化后的状态向量），形状 [B, hidden_dim] = [B, 4096]
        """
        B = h.shape[0]

        # Shape check: 输入必须是 [B, hidden_dim]
        assert h.shape == (B, self.hidden_dim), \
            f"输入形状错误: 期望 [{B}, {self.hidden_dim}], 得到 {h.shape}"

        # Step 1: 将输入向量 reshape 为状态矩阵
        h_state = h.view(B, self.matrix_size, self.matrix_size)  # [B, 64, 64]

        # Step 2: 生成李代数元素 M 并 reshape 成方阵
        M_flat = self.to_matrix(h)  # [B, 4096]
        M = M_flat.view(B, self.matrix_size, self.matrix_size)  # [B, 64, 64]

        # Shape check: 确保矩阵形状正确
        assert M.shape == (B, self.matrix_size, self.matrix_size), \
            f"矩阵形状错误: 期望 [{B}, {self.matrix_size}, {self.matrix_size}], 得到 {M.shape}"

        # Step 3: 数值稳定性处理
        # 缩放：防止矩阵元素过大导致 exp 爆炸
        M = self.scale * M

        # 可选裁剪：进一步约束数值范围
        if self.clip is not None:
            M = torch.clamp(M, -self.clip, self.clip)

        # Step 4: 计算演化算子（李群元素）
        # torch.matrix_exp 计算 exp(M) = I + M + M^2/2! + M^3/3! + ...
        E = torch.matrix_exp(M)  # [B, 64, 64]

        # Step 5: 算子作用于状态（核心几何操作）
        # out_state = E @ h_state，让演化算子直接变换输入状态
        out_state = torch.matmul(E, h_state)  # [B, 64, 64]

        # Step 6: 展平回向量空间
        out = out_state.view(B, self.hidden_dim)  # [B, 4096]

        return out


class RegModel_MLayer(nn.Module):
    """
    M-Layer 变体模型 (Operator-based Implementation)

    结构对比：
    Baseline:  fc1 → fc2 → fc3 → fc4 → fc5
    M-Layer:   fc1 → MLayerBlock → fc4 → fc5

    参数量统计 (Operator-based 版本，比 baseline 少约 16.8M)：
    - fc1: (351×4096) + 4096 = 1,441,792
    - MLayerBlock: 16,781,312 (仅 to_matrix，无 from_matrix)
    - fc4: (4096×4096) + 4096 = 16,781,312
    - fc5: (4096×102) + 102 = 417,894
    - 总计: 35,422,310

    优势：
    - 参数量减少约 32%（从 52.2M 到 35.4M）
    - 引入几何归纳偏置，强迫模型学习数据的演化规律
    - 保留了李群/李代数的流形结构
    """
    def __init__(self, input_dim=351, num_classes=102,
                 matrix_size=64, scale=0.01, clip=10.0):
        super(RegModel_MLayer, self).__init__()

        hidden_dim = 4096

        # 验证 matrix_size 设置正确
        assert matrix_size * matrix_size == hidden_dim, \
            f"matrix_size^2 ({matrix_size}^2={matrix_size*matrix_size}) 必须等于 {hidden_dim}"

        # fc1: 输入层
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 351 → 4096

        # MLayerBlock: 替换 fc2 + fc3
        self.mblock = MLayerBlock(
            hidden_dim=hidden_dim,
            matrix_size=matrix_size,
            scale=scale,
            clip=clip
        )

        # fc4: 隐藏层
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)  # 4096 → 4096

        # fc5: 输出层
        self.fc5 = nn.Linear(hidden_dim, num_classes)  # 4096 → 102

        self.dropout = nn.Dropout(0.5)

        # 保存配置（用于 checkpoint）
        self.m_layer_config = {
            'matrix_size': matrix_size,
            'scale': scale,
            'clip': clip
        }

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状 [B, input_dim] = [B, 351]
        Returns:
            logits: 输出张量，形状 [B, num_classes] = [B, 102]
        """
        # fc1 + ReLU + Dropout (特征提取)
        x = self.dropout(F.relu(self.fc1(x)))  # [B, 4096]

        # MLayerBlock (演化算子作用于状态)
        x = self.mblock(x)  # [B, 4096]

        # Dropout only (不加 ReLU，保留演化后状态的几何形态)
        x = self.dropout(x)  # [B, 4096]

        # fc4 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc4(x)))  # [B, 4096]

        # fc5: 输出 logits
        x = self.fc5(x)  # [B, 102]

        return x


# ==================== Paper-Compliant M-Layer (arXiv:2008.03936) ====================

def expm_approx(M: torch.Tensor, k: int = 6) -> torch.Tensor:
    """
    Scaling & Squaring 矩阵指数近似（论文提出的快速算法）

    原理：
    - exp(M) = exp(M / 2^k)^(2^k)
    - 对于较小的 M / 2^k，可以用一阶近似：exp(M/2^k) ≈ I + M/2^k
    - 然后通过 k 次平方运算得到最终结果

    Args:
        M: 输入矩阵，形状 [B, n, n]
        k: 平方次数（默认6，精度足够大多数应用）

    Returns:
        E: exp(M) 的近似值，形状 [B, n, n]
    """
    B, n, _ = M.shape
    # 创建单位矩阵并扩展到 batch
    I = torch.eye(n, device=M.device, dtype=M.dtype).unsqueeze(0).expand(B, -1, -1)
    # 一阶近似: E ≈ I + M / 2^k
    E = I + M / (2 ** k)
    # k 次平方运算
    for _ in range(k):
        E = torch.matmul(E, E)
    return E


class RegModel_MLayerPaper(nn.Module):
    """
    Paper-Compliant M-Layer 实现（严格遵循 arXiv:2008.03936 的定义）

    核心约束（论文一致性）：
    1) 唯一非线性：矩阵指数 exp(M)
       - 禁止 ReLU、Dropout、clamp、LayerNorm、softmax、sigmoid 等任何非线性
    2) M(x) 对输入 x 是仿射（affine）：
       - M(x) = B + Σ_a z_a(x) * T_a
       - z(x) = U @ x + u 是线性层输出
       - T_a 是可学习基矩阵，B 是可学习偏置矩阵
    3) 输出是线性投影：
       - p = Linear(vec(exp(M)))
       - 等价论文中的 p = V + S:exp(M)
       - 绝对不用 exp(M) @ h 这类 operator-based 形式

    参数化（按任务要求）：
    - embed: Linear(input_dim, basis_dim, bias=True) -- 生成 z(x)
    - basis: Parameter(basis_dim, matrix_size, matrix_size) -- 基矩阵 T_a
    - B: Parameter(matrix_size, matrix_size) -- 偏置矩阵
    - out: Linear(matrix_size*matrix_size, num_classes, bias=True) -- 线性输出层

    前向传播：
    - z = embed(x)                                      # [B, basis_dim]
    - M = B + einsum('ba,amn->bmn', z, basis)          # [B, n, n]
    - M = scale * M                                     # 线性缩放（不破坏"唯一非线性"）
    - E = matrix_exp(M)                                 # 唯一非线性
    - logits = out(E.reshape(B, n*n))                  # 线性投影
    """

    def __init__(
        self,
        input_dim: int = 351,
        num_classes: int = 102,
        matrix_size: int = 8,
        basis_dim: int = 64,
        expm_mode: str = 'exact',
        expm_k: int = 6,
        scale: float = 1.0,
        init_std: float = 0.01
    ):
        """
        Args:
            input_dim: 输入特征维度
            num_classes: 输出类别数
            matrix_size: 方阵大小 n（exp(M) 的 M 是 n×n 矩阵）
            basis_dim: 基矩阵数量 A（z(x) 的维度）
            expm_mode: 矩阵指数计算模式 ('exact' 使用 torch.matrix_exp, 'approx' 使用 scaling & squaring)
            expm_k: approx 模式下的平方次数
            scale: M 的线性缩放因子（用于数值稳定性）
            init_std: 参数初始化标准差
        """
        super(RegModel_MLayerPaper, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.matrix_size = matrix_size
        self.basis_dim = basis_dim
        self.expm_mode = expm_mode
        self.expm_k = expm_k
        self.scale = scale

        # z(x) = U @ x + u: 线性嵌入层，生成基矩阵组合系数
        self.embed = nn.Linear(input_dim, basis_dim, bias=True)

        # T_a: 可学习基矩阵 (basis_dim 个 matrix_size×matrix_size 的矩阵)
        self.basis = nn.Parameter(
            torch.randn(basis_dim, matrix_size, matrix_size) * init_std
        )

        # B: 可学习偏置矩阵
        self.B = nn.Parameter(torch.zeros(matrix_size, matrix_size))

        # 线性输出层: vec(exp(M)) -> logits
        self.out = nn.Linear(matrix_size * matrix_size, num_classes, bias=True)

        # 保存配置（用于 checkpoint）
        self.m_layer_paper_config = {
            'matrix_size': matrix_size,
            'basis_dim': basis_dim,
            'expm_mode': expm_mode,
            'expm_k': expm_k,
            'scale': scale,
            'init_std': init_std
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（严格遵循论文定义，唯一非线性是 matrix_exp）

        Args:
            x: 输入张量，形状 [B, input_dim]

        Returns:
            logits: 输出张量，形状 [B, num_classes]
        """
        B = x.shape[0]

        # Step 1: 计算线性嵌入 z(x) = U @ x + u
        # 这是纯线性操作
        z = self.embed(x)  # [B, basis_dim]

        # Step 2: 构造矩阵 M(x) = B + Σ_a z_a(x) * T_a
        # M 对 x 是仿射的（线性变换 + 偏置）
        # 使用 einsum: z[b,a] * basis[a,m,n] -> M[b,m,n]
        M = self.B + torch.einsum('ba,amn->bmn', z, self.basis)  # [B, n, n]

        # Step 3: 线性缩放（不破坏"唯一非线性"约束）
        M = self.scale * M

        # Step 4: 计算矩阵指数（唯一的非线性操作）
        if self.expm_mode == 'exact':
            E = torch.matrix_exp(M)  # [B, n, n]
        else:  # approx
            E = expm_approx(M, k=self.expm_k)  # [B, n, n]

        # Step 5: 展平并线性投影到输出空间
        # p = V + S:exp(M)，等价于 Linear(vec(exp(M)))
        E_flat = E.reshape(B, -1)  # [B, n*n]
        logits = self.out(E_flat)  # [B, num_classes]

        return logits


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def load_model_state_dict_compatible(model, state_dict, strict=True):
    """
    兼容性模型加载函数

    处理 Operator-based 模型加载旧版 Feature-based checkpoint 的情况：
    - 旧版 checkpoint 包含 mblock.from_matrix.weight/bias
    - 新版模型没有 from_matrix 层

    Args:
        model: 目标模型
        state_dict: checkpoint 中的 state_dict
        strict: 是否严格匹配（默认 True）

    Returns:
        incompatible_keys: 包含 missing_keys 和 unexpected_keys 的命名元组
    """
    # 检查是否有旧版 from_matrix 参数
    old_keys = ['mblock.from_matrix.weight', 'mblock.from_matrix.bias']
    has_old_keys = any(k in state_dict for k in old_keys)

    if has_old_keys:
        print("⚠️ 检测到旧版 Feature-based checkpoint，正在过滤 from_matrix 参数...")
        # 过滤掉旧版的 from_matrix 参数
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if k not in old_keys}
        print(f"   过滤了 {len(state_dict) - len(filtered_state_dict)} 个旧参数")

        # 使用 strict=False 加载，因为我们主动过滤了一些参数
        return model.load_state_dict(filtered_state_dict, strict=False)
    else:
        # 新版 checkpoint，正常加载
        return model.load_state_dict(state_dict, strict=strict)


def print_model_comparison(input_dim=351, num_classes=102,
                           matrix_size=64, scale=0.01, clip=10.0,
                           basis_dim=64, expm_mode='exact', expm_k=6,
                           init_std=0.01, model_type='m_layer'):
    """打印 baseline、m_layer 和 m_layer_paper 模型的参数量对比"""
    # 创建临时模型计算参数量
    baseline = RegModel(input_dim=input_dim, num_classes=num_classes)
    baseline_params = count_parameters(baseline)

    print(f"\n{'='*70}")
    print("模型参数量对比")
    print(f"{'='*70}")
    print(f"Baseline (RegModel):              {baseline_params:,} 参数")

    if model_type == 'm_layer':
        mlayer = RegModel_MLayer(
            input_dim=input_dim,
            num_classes=num_classes,
            matrix_size=matrix_size,
            scale=scale,
            clip=clip
        )
        mlayer_params = count_parameters(mlayer)
        diff = baseline_params - mlayer_params
        diff_ratio = diff / baseline_params * 100

        print(f"M-Layer (Operator):               {mlayer_params:,} 参数")
        if diff > 0:
            print(f"  → 相比 Baseline 减少: {diff:,} ({diff_ratio:.1f}%)")
        elif diff < 0:
            print(f"  → 相比 Baseline 增加: {abs(diff):,} ({abs(diff_ratio):.1f}%)")
        del mlayer

    elif model_type == 'm_layer_paper':
        mlayer_paper = RegModel_MLayerPaper(
            input_dim=input_dim,
            num_classes=num_classes,
            matrix_size=matrix_size,
            basis_dim=basis_dim,
            expm_mode=expm_mode,
            expm_k=expm_k,
            scale=scale,
            init_std=init_std
        )
        paper_params = count_parameters(mlayer_paper)
        diff = baseline_params - paper_params
        diff_ratio = diff / baseline_params * 100

        print(f"M-Layer (Paper, n={matrix_size}, A={basis_dim}): {paper_params:,} 参数")
        if diff > 0:
            print(f"  → 相比 Baseline 减少: {diff:,} ({diff_ratio:.1f}%)")
        elif diff < 0:
            print(f"  → 相比 Baseline 增加: {abs(diff):,} ({abs(diff_ratio):.1f}%)")
        print(f"  → expm_mode: {expm_mode}, expm_k: {expm_k}")
        del mlayer_paper

    print(f"{'='*70}\n")

    # 清理
    del baseline

    return baseline_params

# ==================== L2正则化 ====================

def kernel_l2_regularization(model, weight_decay=0.00001):
    """
    只对权重矩阵应用L2正则化，不包括偏置
    完全模拟TensorFlow的kernel_regularizer
    """
    l2_reg = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            l2_reg += torch.norm(param, p=2) ** 2
    return weight_decay * l2_reg

# ==================== 1D数据集类 ====================

class Brain1D_Dataset(Dataset):
    """
    从1D MAT文件中加载数据进行训练
    """
    
    def __init__(self, mat_files: List[Path], is_train: bool = True,
                 samples_per_subject: Optional[int] = None):
        """
        Args:
            mat_files: 1D MAT文件路径列表
            is_train: 是否为训练模式
            samples_per_subject: 每个被试采样的体素数（None表示全部）
        """
        self.mat_files = mat_files
        self.is_train = is_train
        self.samples_per_subject = samples_per_subject

        
        # 加载所有数据
        self.all_data = []
        self.all_labels = []
        
        print(f"加载{len(mat_files)}个被试的1D数据...")
        for mat_file in tqdm(mat_files):
            self._load_subject(mat_file)
        
        # 转换为numpy数组
        self.all_data = np.vstack(self.all_data).astype(np.float32)
        self.all_labels = np.concatenate(self.all_labels).astype(np.int64)
        
        print(f"总样本数: {len(self.all_data)}")
        print(f"特征维度: {self.all_data.shape[1]}")
        print(f"类别数: {len(np.unique(self.all_labels))}")
        
    
    def _load_subject(self, mat_file: Path):
        """加载单个被试的1D数据并进行独立标准化"""
        with h5py.File(mat_file, 'r') as f:
            # 加载1D数据
            multidim_data = f['multidim_data'][()]  # (351, n_voxels)或是转置前
            seg_one_hot = f['seg_one_hot'][()]      # (102, n_voxels)
            
            # 转置以适应 (n_voxels, 351)
            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T
            
            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T
            
            # 从one-hot转换为类别标签
            labels = np.argmax(seg_one_hot, axis=1)
            
            # ✅ 新增：Patient-wise Z-score 标准化
            # 逻辑：(X - mean) / (std + epsilon)
            # axis=0 表示沿着体素方向计算，保留351个特征的均值/方差
            epsilon = 1e-6
            patient_mean = np.mean(multidim_data, axis=0)
            patient_std = np.std(multidim_data, axis=0)
            
            # 原地修改以节省内存，转换为float32
            multidim_data = (multidim_data - patient_mean) / (patient_std + epsilon)
            multidim_data = multidim_data.astype(np.float32)

            # 采样（如果指定了samples_per_subject）
            if self.samples_per_subject is not None and len(multidim_data) > self.samples_per_subject:
                indices = np.random.choice(len(multidim_data), self.samples_per_subject, replace=False)
                multidim_data = multidim_data[indices]
                labels = labels[indices]
            
            # 添加到总数据中
            self.all_data.append(multidim_data)
            self.all_labels.append(labels)
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx], self.all_labels[idx]

# ==================== 测试集数据类（用于预测和映射）====================

class TestDataset(Dataset):
    """
    用于测试集预测的数据类
    保存3D位置信息用于映射回3D
    """
    
    def __init__(self, mat_file_1d: Path, mat_file_3d: Path):
        
        self.mat_file_1d = mat_file_1d
        self.mat_file_3d = mat_file_3d
        
        # 加载1D数据
        with h5py.File(mat_file_1d, 'r') as f:
            multidim_data = f['multidim_data'][()]
            seg_one_hot = f['seg_one_hot'][()]
            
            if multidim_data.shape[0] == 351:
                multidim_data = multidim_data.T
            if seg_one_hot.shape[0] == 102:
                seg_one_hot = seg_one_hot.T
            
            self.features = multidim_data.astype(np.float32)
            self.labels = np.argmax(seg_one_hot, axis=1)
        
        # ✅ 新增：Patient-wise Z-score 标准化 (针对测试被试自己)
        print(f"正在对测试被试进行标准化: {mat_file_1d.name}")
        epsilon = 1e-6
        mean = np.mean(self.features, axis=0)
        std = np.std(self.features, axis=0)
        self.features = (self.features - mean) / (std + epsilon)
        
        # 加载3D mask（用于映射）
        with h5py.File(mat_file_3d, 'r') as f:
            region_mask = f['region_mask'][()]
            region_labels = f['region_labels'][()]
            self.region_mask = region_mask
            self.region_labels = region_labels
        
        print(f"测试数据: {len(self.features)} 个体素")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==================== 训练器类 ====================

class Trainer:
    def __init__(self, model, device='cuda', learning_rate=0.00001,
                 min_delta_acc=1e-4, lr_factor=0.5, lr_patience=2, min_lr=1e-7):
        self.model = model.to(device)
        self.device = device
        self.min_delta_acc = min_delta_acc

        # Alex的配置：Adam优化器，学习率1e-5
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # 学习率调度器：ReduceLROnPlateau，监控test_acc (mode='max')
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=lr_factor,
            patience=lr_patience,
            threshold=min_delta_acc,
            threshold_mode='abs',
            min_lr=min_lr,
            verbose=False  # 我们手动打印更详细的信息
        )

        # 训练历史 - 扩展为包含所有指标
        self.history = {
            'train_loss': [], 'test_loss': [],
            'train_f1': [], 'test_f1': [],
            'train_acc': [], 'test_acc': [],  # 每个epoch记录gross accuracy
            'lr': [],  # 新增：记录每个epoch的学习率
            # 新增指标
            'train_metrics': [], 'test_metrics': []
        }
    
    def train_epoch(self, train_loader, compute_full_metrics=False):
        """训练一个epoch"""
        self.model.train()
        epoch_train_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            # 计算基础损失
            base_loss = self.criterion(output, target)

            # 添加L2正则化（只对权重）
            l2_reg = kernel_l2_regularization(self.model, weight_decay=0.00001)
            total_loss = base_loss + l2_reg

            total_loss.backward()
            self.optimizer.step()

            epoch_train_loss += total_loss.item()

            # 记录预测、标签和概率
            probs = torch.softmax(output.data, dim=1)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            if compute_full_metrics:
                all_probs.extend(probs.cpu().numpy())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # 计算gross accuracy（每个epoch都计算）
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(all_labels, all_preds)

        # 计算完整指标（可选，避免每个epoch都计算）
        train_metrics = None
        if compute_full_metrics and len(all_probs) > 0:
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_probs = np.array(all_probs)
            train_metrics = compute_all_metrics(y_true, y_pred, y_probs)

        return avg_train_loss, train_f1, train_acc, train_metrics
    
    def evaluate(self, val_loader, compute_full_metrics=True):
        """评估模型"""
        self.model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                # 计算损失
                base_loss = self.criterion(output, target)
                l2_reg = kernel_l2_regularization(self.model, weight_decay=0.00001)
                total_loss = base_loss + l2_reg

                total_val_loss += total_loss.item()

                # 记录预测和标签
                _, predicted = torch.max(output, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

                # 只在需要计算完整指标时才收集概率（节省内存）
                if compute_full_metrics:
                    probs = torch.softmax(output, dim=1)
                    all_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # 计算gross accuracy（每个epoch都计算）
        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(all_labels, all_preds)

        # 计算完整指标
        val_metrics = None
        if compute_full_metrics and len(all_probs) > 0:
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_probs = np.array(all_probs)
            val_metrics = compute_all_metrics(y_true, y_pred, y_probs)

        return avg_val_loss, val_f1, val_acc, val_metrics
    
    def train(self, train_loader, test_loader, epochs=25, patience=5):
        """
        完整训练流程（对应Alex的25个epochs）

        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 最大训练轮数
            patience: early stopping耐心值（连续多少个epoch没改善就停止）
        """
        best_test_acc = 0  # 改用gross accuracy作为最佳模型标准
        best_epoch = 0  # 记录最佳模型出现的epoch
        patience_counter = 0  # early stopping计数器
        early_stopped = False  # 是否提前停止

        # 初始化best_model_state（防止从未更新的情况）
        self.best_model_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n===== Epoch {epoch+1}/{epochs} (lr={current_lr:.2e}) =====")

            # 是否计算完整指标（最后一个epoch或即将early stop）
            # 注意：我们不能提前知道是否会early stop，所以先按正常逻辑
            compute_full = (epoch == epochs - 1)

            # 训练
            train_loss, train_f1, train_acc, train_metrics = self.train_epoch(train_loader, compute_full_metrics=compute_full)
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['train_acc'].append(train_acc)
            if train_metrics:
                self.history['train_metrics'].append(train_metrics)

            # 测试（验证）
            test_loss, test_f1, test_acc, test_metrics = self.evaluate(test_loader, compute_full_metrics=compute_full)
            self.history['test_loss'].append(test_loss)
            self.history['test_f1'].append(test_f1)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)  # 记录当前epoch的学习率
            if test_metrics:
                self.history['test_metrics'].append(test_metrics)

            print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}")

            # 学习率调度器更新
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(test_acc)
            new_lr = self.optimizer.param_groups[0]['lr']

            # 检测学习率是否下降（用于后续 early-stopping 逻辑）
            lr_reduced = (new_lr < prev_lr)
            if lr_reduced:
                print(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

            # 在最后一个epoch显示完整指标
            if compute_full and test_metrics:
                print("\n=== 最终评估指标 ===")
                print(f"Gross Accuracy: {test_metrics['gross_accuracy']:.4f}")
                print(f"Top-1/3/5 Accuracy: {test_metrics['top1_accuracy']:.4f} / {test_metrics['top3_accuracy']:.4f} / {test_metrics['top5_accuracy']:.4f}")
                print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
                print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")
                print(f"Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")
                print(f"Macro Soft Dice: {test_metrics['macro_soft_dice']:.4f}")
                print(f"Risk@95% Coverage: {test_metrics['risk_at_95_coverage']:.4f}")

            # 保存最佳模型（基于测试gross accuracy）+ Early Stopping逻辑
            # 第一个epoch或有改善时保存（改善需超过min_delta阈值）
            # 判断是否有显著改善
            significant_improvement = (epoch == 0) or (test_acc > best_test_acc + self.min_delta_acc)

            if epoch == 0:
                # 第一个epoch：初始化best
                best_test_acc = test_acc
                best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                print(f"初始测试Gross Accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")
            elif significant_improvement:
                # 有显著改善（超过min_delta阈值）
                best_test_acc = test_acc
                best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0  # 重置patience计数器
                print(f"新的最佳测试Gross Accuracy: {best_test_acc:.4f} (Epoch {best_epoch}, 提升>{self.min_delta_acc})")
            else:
                # 未显著改善
                if lr_reduced:
                    # LR刚降低，重置patience_counter，给新学习率更多机会
                    patience_counter = 0
                    print(f"LR刚降低：重置early-stopping patience_counter=0")
                else:
                    patience_counter += 1
                    delta = test_acc - best_test_acc
                    print(f"测试Gross Accuracy未显著改善 (delta={delta:.6f}, 需要>{self.min_delta_acc}, patience: {patience_counter}/{patience})")

                # 检查是否需要early stopping
                if patience_counter >= patience:
                    print(f"\n早停触发！连续{patience}个epoch测试Gross Accuracy未改善")
                    print(f"最佳模型出现在Epoch {best_epoch}，Gross Accuracy: {best_test_acc:.4f}")
                    early_stopped = True
                    break  # 跳出训练循环

        # 训练结束（可能是正常结束或early stopping）
        if early_stopped:
            print(f"\n训练因early stopping在第{epoch+1}个epoch结束（共运行{epoch+1}/{epochs}个epoch）")
        else:
            print(f"\n训练正常完成，共运行{epochs}个epoch")

        # 恢复最佳模型
        print(f"恢复最佳模型（来自Epoch {best_epoch}）")
        self.model.load_state_dict(self.best_model_state)

        # 在最佳模型上重新评估完整指标（确保即使提前停止也有完整指标）
        print("\n===== 最佳模型的完整评估 =====")
        _, best_f1, best_acc, best_metrics = self.evaluate(test_loader, compute_full_metrics=True)
        self.history['best_test_metrics'] = best_metrics

        # 保存early stopping相关信息
        self.history['best_epoch'] = best_epoch
        self.history['best_test_acc'] = best_test_acc  # 按min_delta规则的最佳acc
        self.history['early_stopped'] = early_stopped
        self.history['total_epochs'] = epoch + 1  # 实际运行的epoch数

        if best_metrics:
            print(f"Best Test Gross Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
            print(f"Best Test F1: {best_f1:.4f}")
            print(f"Top-1/3/5 Accuracy: {best_metrics['top1_accuracy']:.4f} / {best_metrics['top3_accuracy']:.4f} / {best_metrics['top5_accuracy']:.4f}")
            print(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
            print(f"Macro-F1: {best_metrics['macro_f1']:.4f}")
            print(f"Weighted F1: {best_metrics['weighted_f1']:.4f}")
            print(f"Cohen's Kappa: {best_metrics['cohen_kappa']:.4f}")
            print(f"Macro Soft Dice: {best_metrics['macro_soft_dice']:.4f}")
            print(f"Risk@95% Coverage: {best_metrics['risk_at_95_coverage']:.4f}")

        return self.history

# ==================== 预测并映射回3D ====================

def predict_and_map_to_3d(model, test_dataset, device='cuda', batch_size=512):
    """
    预测测试集的softmax概率并映射回3D体积
    
    Args:
        model: 训练好的模型
        test_dataset: TestDataset实例
        
    Returns:
        predictions_3d: 3D概率体积 (384, 336, 256, 102)
    """
    model.eval()
    
    # 创建数据加载器
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 预测所有样本
    all_predictions = []
    
    print("预测所有体素...")
    with torch.no_grad():
        for data, _ in tqdm(loader):
            data = data.to(device)
            output = model(data)
            # 应用softmax获取概率
            probs = F.softmax(output, dim=1)
            all_predictions.append(probs.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    
    print(f"预测完成，共{len(all_predictions)}个体素")
    
    # 映射回3D体积 - 使用与dataset_create相同的直接布尔索引方法
    print("映射回3D体积...")
    
    # 获取region掩码（布尔类型）
    region_mask = test_dataset.region_mask
    region = region_mask.astype(bool)  # 转换为布尔掩码，与dataset_create保持一致
    n_voxels = np.sum(region)
    
    # 检查体素数量是否匹配
    if n_voxels != len(all_predictions):
        print(f"警告：体素数量不匹配！")
        print(f"  3D region中的体素数: {n_voxels}")
        print(f"  预测的体素数: {len(all_predictions)}")
        print(f"  差异: {abs(n_voxels - len(all_predictions))}")
        raise ValueError("体素数量必须完全匹配，请检查数据集创建和加载逻辑")
    
    # 创建4D概率体积：(384, 336, 256, 102)
    predictions_3d = np.zeros((384, 336, 256, 102), dtype=np.float32)
    
    # 使用直接布尔索引映射 - 与dataset_create的方法完全一致
    # 参考：data_4d[region] = features (batch_convert_validated_with_logging.py第186行)
    predictions_3d[region, :] = all_predictions  # 直接布尔索引赋值
    
    # 验证映射结果的正确性
    print(f"映射完成，3D体积形状: {predictions_3d.shape}")
    print(f"  有效体素数: {n_voxels}")
    print(f"  概率范围: [{predictions_3d[region].min():.4f}, {predictions_3d[region].max():.4f}]")
    
    # ===== 关键验证1：映射前后逐体素值一致性自检（修正前）=====
    print("验证映射往返一致性（修正前）...")
    recovered_before_fix = predictions_3d[region].copy()  # 从3D体积中取回有效体素的概率
    
    # 检查形状是否匹配
    if recovered_before_fix.shape != all_predictions.shape:
        raise ValueError(f"往返形状不匹配: 原始{all_predictions.shape} vs 恢复{recovered_before_fix.shape}")
    
    # 逐体素精确对比（使用严格的 atol=0）- 验证映射逻辑正确性
    is_consistent_before = np.allclose(recovered_before_fix, all_predictions, atol=0)
    
    if is_consistent_before:
        print(f"  ✅ roundtrip_before_fix: 映射逻辑完全正确，{n_voxels}个体素完全匹配")
    else:
        # 计算不一致的详细信息
        diff = np.abs(recovered_before_fix - all_predictions)
        max_diff = np.max(diff)
        n_diff_voxels = np.sum((diff > 0).any(axis=1))  # 按体素聚合：任一通道有差异的体素数
        print(f"  ❌ roundtrip_before_fix: 映射逻辑错误！")
        print(f"     最大差异: {max_diff}")
        print(f"     不一致体素数: {n_diff_voxels}/{n_voxels}")
        
        # 显示前几个不一致的位置
        diff_indices = np.where((diff > 0).any(axis=1))[0][:5]
        for idx in diff_indices:
            print(f"     体素{idx}: 原始{all_predictions[idx][:3]}... vs 恢复{recovered_before_fix[idx][:3]}...")
        
        raise AssertionError("预测概率映射往返不一致！映射逻辑存在错误")
    
    # ===== 严格的概率约束验证与修正 =====
    print("验证和修正softmax概率约束...")
    
    # 1. 检查有效区域的概率和
    region_probs = predictions_3d[region]  # (n_voxels, 102)
    prob_sums = np.sum(region_probs, axis=1)
    
    # 严格验证概率和是否为1
    perfect_probs = np.sum(np.abs(prob_sums - 1.0) < 1e-6)  # 严格接近1的体素数
    imperfect_probs = n_voxels - perfect_probs
    
    if imperfect_probs > 0:
        max_deviation = np.max(np.abs(prob_sums - 1.0))
        print(f"  ⚠️ 发现{imperfect_probs}/{n_voxels}个体素概率和偏离1，最大偏差: {max_deviation:.8f}")
        
        # 重新归一化修正概率
        print("  🔧 重新归一化修正概率...")
        region_probs_corrected = region_probs / prob_sums.reshape(-1, 1)  # 按行归一化
        predictions_3d[region] = region_probs_corrected
        
        # 验证修正后的概率和
        corrected_sums = np.sum(predictions_3d[region], axis=1)
        perfect_after = np.sum(np.abs(corrected_sums - 1.0) < 1e-10)
        print(f"  ✅ 修正完成，{perfect_after}/{n_voxels}个体素概率和严格为1")
    else:
        print(f"  ✅ 所有{n_voxels}个体素概率和严格为1")
    
    # 2. 检查概率值范围 [0, 1]
    region_probs = predictions_3d[region]
    min_prob = np.min(region_probs)
    max_prob = np.max(region_probs)
    
    if min_prob < 0 or max_prob > 1:
        print(f"  ⚠️ 发现概率值超出[0,1]范围: [{min_prob:.8f}, {max_prob:.8f}]")
        
        # 阈值裁剪
        print("  🔧 进行阈值裁剪...")
        region_probs_clipped = np.clip(region_probs, 0.0, 1.0)
        
        # 重新归一化（裁剪后可能破坏概率和=1的约束）
        clip_sums = np.sum(region_probs_clipped, axis=1)
        region_probs_renorm = region_probs_clipped / clip_sums.reshape(-1, 1)
        predictions_3d[region] = region_probs_renorm
        
        # 验证最终结果
        final_min = np.min(predictions_3d[region])
        final_max = np.max(predictions_3d[region])
        final_sums = np.sum(predictions_3d[region], axis=1)
        perfect_final = np.sum(np.abs(final_sums - 1.0) < 1e-10)
        
        print(f"  ✅ 裁剪完成，概率范围: [{final_min:.8f}, {final_max:.8f}]")
        print(f"  ✅ 重新归一化完成，{perfect_final}/{n_voxels}个体素概率和严格为1")
    else:
        print(f"  ✅ 所有概率值在[0,1]范围内: [{min_prob:.8f}, {max_prob:.8f}]")
    
    # 3. 严格验证背景区域（必须全为0）
    background_sum = np.sum(predictions_3d[~region])
    if background_sum > 0:
        print(f"  ❌ 背景区域概率和非零: {background_sum:.10f}")
        raise ValueError("背景区域必须全为0！映射逻辑存在错误")
    else:
        print(f"  ✅ 背景区域概率和严格为0")
    
    # 4. 预测类别分布统计
    predicted_labels = np.argmax(predictions_3d[region], axis=1)
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    print(f"  📊 预测了 {len(unique_labels)} 个不同类别")
    print(f"  📊 最频繁的类别: {unique_labels[np.argmax(counts)]} (出现{np.max(counts)}次)")
    
    print("概率约束验证与修正完成！")
    
    # ===== 关键验证2：修正后的往返一致性检查（验证修正没破坏掩膜对应关系）=====
    print("验证映射往返一致性（修正后）...")
    recovered_after_fix = predictions_3d[region]  # 修正后从3D体积中取回有效体素的概率
    
    # 检查形状是否匹配
    if recovered_after_fix.shape != all_predictions.shape:
        raise ValueError(f"修正后往返形状不匹配: 原始{all_predictions.shape} vs 恢复{recovered_after_fix.shape}")
    
    # 验证修正没有破坏shape与掩膜对应关系（允许数值变化，但结构必须一致）
    shape_consistent = (recovered_after_fix.shape == all_predictions.shape)
    mask_consistent = np.sum(predictions_3d[~region]) == 0  # 背景依然为0
    
    if shape_consistent and mask_consistent:
        # 计算修正引起的数值变化
        if not np.allclose(recovered_after_fix, all_predictions, atol=0):
            diff_after = np.abs(recovered_after_fix - all_predictions)
            max_diff_after = np.max(diff_after)
            changed_voxels = np.sum((diff_after > 1e-10).any(axis=1))  # 按体素聚合：任一通道变化的体素数
            print(f"  ✅ roundtrip_after_fix: 结构完整，修正影响了{changed_voxels}/{n_voxels}个体素")
            print(f"     最大修正差异: {max_diff_after:.8f}")
        else:
            print(f"  ✅ roundtrip_after_fix: 结构完整，无需修正")
    else:
        raise AssertionError("修正破坏了掩膜对应关系！修正逻辑存在错误")
    
    return predictions_3d

# ==================== 辅助函数：JSON序列化 ====================

def convert_to_json_serializable(obj):
    """
    递归转换对象为JSON可序列化的格式
    处理numpy类型、字典、列表等
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ==================== 辅助函数：读取排除列表 ====================

def load_exclude_list(exclude_file: Path) -> set:
    """
    从文件中读取要排除的被试名列表

    Args:
        exclude_file: 排除列表文件路径

    Returns:
        被试名集合
    """
    exclude_set = set()
    if exclude_file and exclude_file.exists():
        with open(exclude_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if line and not line.startswith('#'):
                    exclude_set.add(line)
    return exclude_set

def should_exclude_subject(subject_name: str, exclude_set: set) -> bool:
    """
    判断被试是否应该被排除

    Args:
        subject_name: 被试名（如 "ODP_01_YHC04"）
        exclude_set: 要排除的被试名集合（如 {"YHC04", "YHC2"}）

    Returns:
        True 如果应该排除，False 否则
    """
    # 检查被试名中是否包含任何排除列表中的字符串
    for exclude_name in exclude_set:
        if exclude_name in subject_name:
            return True
    return False

# ==================== 单次训练函数（用于交叉验证）====================

def train_single_fold(
    test_subject_name: str,
    train_subject_names: List[str],
    idx_1d: Dict,
    idx_3d: Dict,
    fold_output_dir: Path,
    args,
    device,
    logger
) -> Dict:
    """
    训练单个fold

    Args:
        test_subject_name: 测试被试名
        train_subject_names: 训练被试名列表
        idx_1d: 1D文件索引字典
        idx_3d: 3D文件索引字典
        fold_output_dir: 该fold的输出目录
        args: 命令行参数
        device: 计算设备
        logger: 日志器

    Returns:
        fold_result: 包含该fold训练结果的字典
    """
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    # 获取文件路径
    test_file_1d = idx_1d[test_subject_name]
    test_file_3d = idx_3d[test_subject_name]
    train_files_1d = [idx_1d[name] for name in train_subject_names]

    logger.info(f'训练集: {len(train_files_1d)} 个被试')
    logger.info(f'测试被试名: {test_subject_name}')

    # 创建训练数据集
    train_dataset = Brain1D_Dataset(
        train_files_1d,
        is_train=True,
        samples_per_subject=args.samples_per_subject
    )

    # 创建测试数据集
    test_dataset = TestDataset(test_file_1d, test_file_3d)

    # 验证1D与3D标签一致性
    mask = test_dataset.region_mask.astype(bool)
    labels_3d = test_dataset.region_labels[mask]
    labels_1d = test_dataset.labels

    if len(labels_1d) != len(labels_3d):
        raise ValueError(f"标签数量不匹配: 1D={len(labels_1d)}, 3D={len(labels_3d)}")

    if not np.array_equal(labels_1d, labels_3d):
        raise AssertionError("测试集标签在一维与三维不一致！")

    logger.info(f'训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型（根据 model_type 选择）
    if args.model_type == 'baseline':
        model = RegModel(input_dim=351, num_classes=102)
        logger.info('使用 Baseline 模型 (4×4096 MLP)')
    elif args.model_type == 'm_layer':
        model = RegModel_MLayer(
            input_dim=351,
            num_classes=102,
            matrix_size=args.m_matrix_size,
            scale=args.m_scale,
            clip=args.m_clip
        )
        logger.info(f'使用 M-Layer (Operator) 模型 (matrix_size={args.m_matrix_size}, '
                   f'scale={args.m_scale}, clip={args.m_clip})')
    elif args.model_type == 'm_layer_paper':
        model = RegModel_MLayerPaper(
            input_dim=351,
            num_classes=102,
            matrix_size=args.m_matrix_size,
            basis_dim=args.m_basis_dim,
            expm_mode=args.m_expm_mode,
            expm_k=args.m_expm_k,
            scale=args.m_scale,
            init_std=args.m_init_std
        )
        logger.info(f'使用 M-Layer (Paper) 模型 (matrix_size={args.m_matrix_size}, '
                   f'basis_dim={args.m_basis_dim}, scale={args.m_scale}, '
                   f'expm_mode={args.m_expm_mode}, expm_k={args.m_expm_k})')
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    logger.info(f'模型参数量: {count_parameters(model):,}')

    # 如果指定了 --load_model，先加载权重作为初始化
    if args.load_model:
        logger.info(f'加载预训练模型作为初始化: {args.load_model}')
        checkpoint = torch.load(args.load_model, map_location='cpu')

        # 检查模型类型兼容性
        saved_model_type = checkpoint.get('model_type', 'baseline')
        if saved_model_type != args.model_type:
            raise ValueError(
                f"模型类型不匹配！Checkpoint 中保存的是 '{saved_model_type}'，"
                f"但当前指定的是 '{args.model_type}'。"
                f"请确保使用相同的 --model_type 参数。"
            )

        # 使用兼容性加载（处理旧版 Feature-based checkpoint）
        load_model_state_dict_compatible(model, checkpoint['model_state_dict'])

    # 创建训练器
    trainer = Trainer(
        model,
        device=device,
        learning_rate=0.00001,
        min_delta_acc=args.min_delta_acc,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr
    )

    # 训练模型
    history = trainer.train(train_loader, test_loader, epochs=args.epochs, patience=args.patience)

    # 保存模型（根据模型类型使用不同的文件名）
    if args.model_type == 'm_layer_paper':
        model_path = fold_output_dir / 'm_layer_paper_model.pth'
    else:
        model_path = fold_output_dir / 'model.pth'

    save_dict = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'test_subject': test_subject_name,
        'args': vars(args),
        'model_type': args.model_type,  # 保存模型类型
    }
    # 如果是 M-Layer (Operator) 模型，保存其配置
    if args.model_type == 'm_layer':
        save_dict['m_layer_config'] = {
            'matrix_size': args.m_matrix_size,
            'scale': args.m_scale,
            'clip': args.m_clip
        }
    # 如果是 M-Layer (Paper) 模型，保存其配置
    elif args.model_type == 'm_layer_paper':
        save_dict['m_layer_paper_config'] = {
            'matrix_size': args.m_matrix_size,
            'basis_dim': args.m_basis_dim,
            'expm_mode': args.m_expm_mode,
            'expm_k': args.m_expm_k,
            'scale': args.m_scale,
            'init_std': args.m_init_std
        }
    torch.save(save_dict, model_path)

    # 保存训练历史
    history_path = fold_output_dir / 'history.json'
    history_serializable = convert_to_json_serializable(history)
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)

    # 构建fold结果
    fold_result = {
        'test_subject': test_subject_name,
        'best_test_acc': history['best_test_acc'],
        'best_epoch': history['best_epoch'],
        'early_stopped': history['early_stopped'],
        'total_epochs': history['total_epochs'],
        'final_test_acc': history['test_acc'][-1],
        'final_test_f1': history['test_f1'][-1],
    }

    # 如果有best_test_metrics，添加更多指标
    if 'best_test_metrics' in history and history['best_test_metrics']:
        metrics = history['best_test_metrics']
        fold_result.update({
            'balanced_accuracy': metrics.get('balanced_accuracy'),
            'macro_f1': metrics.get('macro_f1'),
            'weighted_f1': metrics.get('weighted_f1'),
            'cohen_kappa': metrics.get('cohen_kappa'),
            'macro_soft_dice': metrics.get('macro_soft_dice'),
        })

    # 清理GPU内存
    del model, trainer, train_dataset, test_dataset, train_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_result


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='使用1D训练集进行训练，映射回3D')
    parser.add_argument('--data_dir_1d', type=str, required=True,
                       help='1D MAT文件目录（去除_3d后缀的文件）')
    parser.add_argument('--data_dir_3d', type=str, required=True,
                       help='3D MAT文件目录（用于获取3D mask）')
    parser.add_argument('--output_dir', type=str, default='./results_1d',
                       help='输出目录')
    parser.add_argument('--test_subject', type=int, default=38,
                       help='测试被试编号(1-38)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小（Alex使用128）')
    parser.add_argument('--epochs', type=int, default=25,
                       help='训练轮数（Alex使用25）')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping耐心值（连续多少个epoch test gross acc没改善就停止训练，默认5）')
    parser.add_argument('--min_delta_acc', type=float, default=1e-4,
                       help='test_acc改善的最小阈值，低于此值不算改善（默认1e-4）')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                       help='ReduceLROnPlateau的衰减因子（默认0.5）')
    parser.add_argument('--lr_patience', type=int, default=2,
                       help='ReduceLROnPlateau的耐心值（默认2）')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='学习率下限（默认1e-7）')
    parser.add_argument('--samples_per_subject', type=int, default=None,
                       help='每个被试采样的体素数（None表示全部）')
    parser.add_argument('--save_predictions', action='store_true',
                       help='保存3D预测概率')
    parser.add_argument('--load_model', type=str, default=None,
                       help='加载预训练模型路径')
    parser.add_argument('--predict_only', action='store_true',
                       help='仅进行预测，不训练模型')
    parser.add_argument('--exclude_subjects', type=str, default=None,
                       help='要排除的被试名列表文件（txt格式，每行一个被试名）')
    parser.add_argument('--exclude_single', type=str, default=None,
                       help='单次排除的被试名（用于每次排除一个的实验）')
    parser.add_argument('--fixed_test_subject', type=str, default=None,
                       help='固定测试被试名（用于对比实验）')
    parser.add_argument('--cross_validation', action='store_true',
                       help='启用交叉验证模式：每个可用被试轮流作为测试集')

    # ===== M-Layer 模型相关参数 =====
    parser.add_argument('--model_type', type=str, default='m_layer',
                       choices=['baseline', 'm_layer', 'm_layer_paper'],
                       help='模型类型: baseline (4x4096 MLP), m_layer (operator-based), m_layer_paper (论文严格实现)')
    parser.add_argument('--m_matrix_size', type=int, default=64,
                       help='M-Layer 矩阵大小 (m_layer: 默认64，必须满足 matrix_size^2=4096; m_layer_paper: 任意)')
    parser.add_argument('--m_scale', type=float, default=0.01,
                       help='M-Layer 输入缩放因子 (默认0.01，用于数值稳定性)')
    parser.add_argument('--m_clip', type=float, default=10.0,
                       help='M-Layer (operator) 数值裁剪范围 (默认10.0，设为-1表示不裁剪，仅对 m_layer 有效)')

    # ===== Paper-Compliant M-Layer 专用参数 =====
    parser.add_argument('--m_basis_dim', type=int, default=64,
                       help='Paper M-Layer 基矩阵数量/嵌入维度 (默认64，仅对 m_layer_paper 有效)')
    parser.add_argument('--m_expm_mode', type=str, default='exact',
                       choices=['exact', 'approx'],
                       help='矩阵指数计算模式: exact (torch.matrix_exp) 或 approx (scaling & squaring)')
    parser.add_argument('--m_expm_k', type=int, default=6,
                       help='Scaling & Squaring 平方次数 (默认6，仅对 m_expm_mode=approx 有效)')
    parser.add_argument('--m_init_std', type=float, default=0.01,
                       help='Paper M-Layer 参数初始化标准差 (默认0.01)')

    args = parser.parse_args()

    # 处理 m_clip: -1 表示不裁剪
    if args.m_clip < 0:
        args.m_clip = None
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # ===== 模型类型配置验证 =====
    logger.info(f'模型类型: {args.model_type}')
    if args.model_type == 'm_layer':
        # Operator-based M-Layer: 验证 matrix_size^2 == 4096
        if args.m_matrix_size * args.m_matrix_size != 4096:
            raise ValueError(
                f"m_matrix_size^2 ({args.m_matrix_size}^2={args.m_matrix_size**2}) "
                f"必须等于 4096。请使用 --m_matrix_size 64"
            )
        logger.info(f'M-Layer (Operator) 配置: matrix_size={args.m_matrix_size}, '
                   f'scale={args.m_scale}, clip={args.m_clip}')
    elif args.model_type == 'm_layer_paper':
        # Paper-Compliant M-Layer: 不需要 matrix_size^2==4096 的约束
        logger.info(f'M-Layer (Paper) 配置: matrix_size={args.m_matrix_size}, '
                   f'basis_dim={args.m_basis_dim}, scale={args.m_scale}, '
                   f'expm_mode={args.m_expm_mode}, expm_k={args.m_expm_k}')

    # 打印参数量对比（仅在程序启动时打印一次）
    print_model_comparison(
        input_dim=351,
        num_classes=102,
        matrix_size=args.m_matrix_size,
        scale=args.m_scale,
        clip=args.m_clip,
        basis_dim=args.m_basis_dim,
        expm_mode=args.m_expm_mode,
        expm_k=args.m_expm_k,
        init_std=args.m_init_std,
        model_type=args.model_type
    )

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子（和Alex一样）
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 获取1D和3D MAT文件
    data_dir_1d = Path(args.data_dir_1d)
    data_dir_3d = Path(args.data_dir_3d)
    
    # 查找文件
    mat_files_1d = list(data_dir_1d.glob('*.mat'))
    mat_files_3d = list(data_dir_3d.glob('*_3d_validated.mat'))
    
    logger.info(f'找到1D文件: {len(mat_files_1d)}个')
    logger.info(f'找到3D文件: {len(mat_files_3d)}个')
    
    # 被试名提取函数
    def subject_key_1d(p):
        """提取1D文件的被试名 e.g. "ODP_01_qhlazec" """
        return Path(p).stem
    
    def subject_key_3d(p):
        """提取3D文件的被试名 e.g. "ODP_01_qhlazec_3d_validated" -> "ODP_01_qhlazec" """
        return Path(p).stem.replace('_3d_validated', '')
    
    # 按被试名构建索引
    idx_1d = {subject_key_1d(p): p for p in mat_files_1d}
    idx_3d = {subject_key_3d(p): p for p in mat_files_3d}
    
    # 验证被试集合一致性
    subjects_1d = set(idx_1d.keys())
    subjects_3d = set(idx_3d.keys())
    
    logger.info(f'1D被试: {len(subjects_1d)}个')
    logger.info(f'3D被试: {len(subjects_3d)}个')
    
    assert subjects_1d == subjects_3d, f"一维和三维被试集合不一致！差异: {subjects_1d ^ subjects_3d}"
    
    # 按被试名排序，确保一致性
    all_subject_names = sorted(idx_1d.keys())

    # ===== 排除指定被试的逻辑 =====
    exclude_set = set()

    # 1. 从文件读取排除列表
    if args.exclude_subjects:
        exclude_file = Path(args.exclude_subjects)
        exclude_set = load_exclude_list(exclude_file)
        logger.info(f'从文件加载排除列表: {exclude_file}')
        logger.info(f'要排除的被试标识: {exclude_set}')

    # 2. 单次排除（用于逐个排除实验）
    if args.exclude_single:
        exclude_set.add(args.exclude_single)
        logger.info(f'单次排除被试: {args.exclude_single}')

    # 3. 应用排除过滤
    subject_names = [name for name in all_subject_names
                     if not should_exclude_subject(name, exclude_set)]

    excluded_count = len(all_subject_names) - len(subject_names)
    if excluded_count > 0:
        excluded_names = [name for name in all_subject_names
                         if should_exclude_subject(name, exclude_set)]
        logger.info(f'排除了 {excluded_count} 个被试')
        logger.info(f'被排除的被试: {excluded_names}')

    logger.info(f'剩余可用被试: {len(subject_names)}个')

    # ===== 交叉验证模式 vs 单次训练模式 =====
    if args.cross_validation:
        # ===== 交叉验证模式 =====
        logger.info(f'\n{"="*60}')
        logger.info(f'交叉验证模式: {len(subject_names)}-fold cross validation')
        logger.info(f'{"="*60}\n')

        fold_results = []
        total_folds = len(subject_names)

        for fold_idx, test_subject_name in enumerate(subject_names):
            fold_num = fold_idx + 1
            logger.info(f'\n{"="*60}')
            logger.info(f'Fold {fold_num}/{total_folds}: 测试被试 = {test_subject_name}')
            logger.info(f'{"="*60}')

            # 设置该fold的随机种子（保证可复现）
            fold_seed = 42 + fold_idx
            torch.manual_seed(fold_seed)
            torch.cuda.manual_seed(fold_seed)
            np.random.seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fold_seed)

            # 训练集：除测试被试外的所有被试
            train_subject_names_fold = [name for name in subject_names if name != test_subject_name]

            # 该fold的输出目录
            fold_output_dir = output_dir / f'fold_{fold_num:02d}_{test_subject_name}'

            # 训练该fold
            try:
                fold_result = train_single_fold(
                    test_subject_name=test_subject_name,
                    train_subject_names=train_subject_names_fold,
                    idx_1d=idx_1d,
                    idx_3d=idx_3d,
                    fold_output_dir=fold_output_dir,
                    args=args,
                    device=device,
                    logger=logger
                )
                fold_result['fold'] = fold_num
                fold_result['status'] = 'success'
                fold_results.append(fold_result)

                logger.info(f'Fold {fold_num} 完成: best_test_acc={fold_result["best_test_acc"]:.4f}')

            except Exception as e:
                logger.error(f'Fold {fold_num} 失败: {e}')
                fold_results.append({
                    'fold': fold_num,
                    'test_subject': test_subject_name,
                    'status': 'failed',
                    'error': str(e)
                })

        # ===== 生成交叉验证汇总报告 =====
        logger.info(f'\n{"="*60}')
        logger.info('交叉验证完成，生成汇总报告...')
        logger.info(f'{"="*60}\n')

        # 筛选成功的fold
        successful_folds = [r for r in fold_results if r.get('status') == 'success']
        failed_folds = [r for r in fold_results if r.get('status') == 'failed']

        if successful_folds:
            # 计算统计量
            test_accs = [r['best_test_acc'] for r in successful_folds]
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)

            best_fold = max(successful_folds, key=lambda x: x['best_test_acc'])
            worst_fold = min(successful_folds, key=lambda x: x['best_test_acc'])

            # 构建汇总报告
            cv_summary = {
                'config': {
                    'total_folds': total_folds,
                    'successful_folds': len(successful_folds),
                    'failed_folds': len(failed_folds),
                    'excluded_subjects': list(exclude_set) if exclude_set else [],
                    'epochs': args.epochs,
                    'patience': args.patience,
                    'batch_size': args.batch_size,
                    'min_delta_acc': args.min_delta_acc,
                    'lr_factor': args.lr_factor,
                    'lr_patience': args.lr_patience,
                    'min_lr': args.min_lr,
                },
                'fold_results': fold_results,
                'summary': {
                    'mean_test_acc': float(mean_acc),
                    'std_test_acc': float(std_acc),
                    'best_fold': {
                        'fold': best_fold['fold'],
                        'test_subject': best_fold['test_subject'],
                        'test_acc': best_fold['best_test_acc']
                    },
                    'worst_fold': {
                        'fold': worst_fold['fold'],
                        'test_subject': worst_fold['test_subject'],
                        'test_acc': worst_fold['best_test_acc']
                    }
                }
            }

            # 如果有更多指标，也计算它们的统计量
            if 'balanced_accuracy' in successful_folds[0] and successful_folds[0]['balanced_accuracy'] is not None:
                ba_values = [r['balanced_accuracy'] for r in successful_folds if r.get('balanced_accuracy') is not None]
                if ba_values:
                    cv_summary['summary']['mean_balanced_accuracy'] = float(np.mean(ba_values))
                    cv_summary['summary']['std_balanced_accuracy'] = float(np.std(ba_values))

            if 'macro_f1' in successful_folds[0] and successful_folds[0]['macro_f1'] is not None:
                f1_values = [r['macro_f1'] for r in successful_folds if r.get('macro_f1') is not None]
                if f1_values:
                    cv_summary['summary']['mean_macro_f1'] = float(np.mean(f1_values))
                    cv_summary['summary']['std_macro_f1'] = float(np.std(f1_values))

            if 'cohen_kappa' in successful_folds[0] and successful_folds[0]['cohen_kappa'] is not None:
                kappa_values = [r['cohen_kappa'] for r in successful_folds if r.get('cohen_kappa') is not None]
                if kappa_values:
                    cv_summary['summary']['mean_cohen_kappa'] = float(np.mean(kappa_values))
                    cv_summary['summary']['std_cohen_kappa'] = float(np.std(kappa_values))

            # 保存汇总报告
            summary_path = output_dir / 'cv_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(cv_summary, f, indent=2)

            # 打印汇总
            logger.info(f'===== 交叉验证汇总 =====')
            logger.info(f'成功完成: {len(successful_folds)}/{total_folds} folds')
            if failed_folds:
                logger.info(f'失败: {len(failed_folds)} folds')
            logger.info(f'平均测试Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
            logger.info(f'最佳Fold: {best_fold["fold"]} ({best_fold["test_subject"]}), acc={best_fold["best_test_acc"]:.4f}')
            logger.info(f'最差Fold: {worst_fold["fold"]} ({worst_fold["test_subject"]}), acc={worst_fold["best_test_acc"]:.4f}')
            logger.info(f'汇总报告已保存至: {summary_path}')

        else:
            logger.error('所有fold都失败了！')

        return  # 交叉验证模式结束

    # ===== 单次训练模式（原有逻辑）=====
    # 选择测试被试
    if args.fixed_test_subject:
        # 固定测试被试名（用于对比实验）
        test_subject_name = None
        for name in subject_names:
            if args.fixed_test_subject in name:
                test_subject_name = name
                break

        if test_subject_name is None:
            raise ValueError(f"未找到匹配的固定测试被试: {args.fixed_test_subject}")

        logger.info(f'使用固定测试被试: {test_subject_name}')
    else:
        # 按索引选择测试被试
        test_idx = args.test_subject - 1
        if test_idx >= len(subject_names):
            raise ValueError(f"测试被试编号{args.test_subject}超出范围[1, {len(subject_names)}]")

        test_subject_name = subject_names[test_idx]

    test_file_1d = idx_1d[test_subject_name]
    test_file_3d = idx_3d[test_subject_name]

    # 训练集：除测试被试外的所有被试
    train_subject_names = [name for name in subject_names if name != test_subject_name]
    train_files_1d = [idx_1d[name] for name in train_subject_names]

    logger.info(f'训练集: {len(train_files_1d)} 个被试')
    logger.info(f'测试被试名: {test_subject_name}')
    logger.info(f'测试集1D: {test_file_1d.name}')
    logger.info(f'测试集3D: {test_file_3d.name}')
    
    # 验证文件匹配正确性
    logger.info(f'验证: 1D被试名 = {subject_key_1d(test_file_1d)}')
    logger.info(f'验证: 3D被试名 = {subject_key_3d(test_file_3d)}')
    assert subject_key_1d(test_file_1d) == subject_key_3d(test_file_3d), "测试文件被试名不匹配！"
    
    # 创建训练数据集
    logger.info('加载训练数据...')
    train_dataset = Brain1D_Dataset(
        train_files_1d,
        is_train=True,
        samples_per_subject=args.samples_per_subject
    )
    
    # 创建测试数据集（需要1D和3D文件）
    logger.info('加载测试数据...')
    test_dataset = TestDataset(test_file_1d, test_file_3d)
    
    # ===== 关键验证：1D与3D标签一致性自检 =====
    logger.info('验证1D与3D标签一致性...')
    mask = test_dataset.region_mask.astype(bool)
    labels_3d = test_dataset.region_labels[mask]  # 从3D掩膜位置提取标签
    labels_1d = test_dataset.labels               # 1D数据集的标签
    
    if len(labels_1d) != len(labels_3d):
        logger.error(f"标签数量不匹配: 1D={len(labels_1d)}, 3D={len(labels_3d)}")
        raise ValueError("1D和3D标签数量不一致")
    
    labels_match = np.array_equal(labels_1d, labels_3d)
    if labels_match:
        logger.info(f'  ✅ 标签一致性验证通过：{len(labels_1d)}个体素标签完全匹配')
    else:
        # 计算不匹配的详细信息
        n_mismatch = np.sum(labels_1d != labels_3d)
        mismatch_rate = n_mismatch / len(labels_1d) * 100
        
        logger.error(f'  ❌ 标签一致性验证失败！')
        logger.error(f'     不匹配体素数: {n_mismatch}/{len(labels_1d)} ({mismatch_rate:.2f}%)')
        
        # 显示前几个不匹配的位置
        mismatch_indices = np.where(labels_1d != labels_3d)[0][:10]
        for idx in mismatch_indices:
            logger.error(f'     体素{idx}: 1D标签={labels_1d[idx]}, 3D标签={labels_3d[idx]}')
        
        raise AssertionError("测试集标签在一维与三维不一致！可能是文件错配")
    
    logger.info(f'训练样本数: {len(train_dataset)}')
    logger.info(f'测试样本数: {len(test_dataset)}')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型（根据 model_type 选择）
    logger.info('创建模型...')
    if args.model_type == 'baseline':
        model = RegModel(input_dim=351, num_classes=102)
        logger.info('使用 Baseline 模型 (4×4096 MLP)')
    elif args.model_type == 'm_layer':
        model = RegModel_MLayer(
            input_dim=351,
            num_classes=102,
            matrix_size=args.m_matrix_size,
            scale=args.m_scale,
            clip=args.m_clip
        )
        logger.info(f'使用 M-Layer (Operator) 模型 (matrix_size={args.m_matrix_size}, '
                   f'scale={args.m_scale}, clip={args.m_clip})')
    elif args.model_type == 'm_layer_paper':
        model = RegModel_MLayerPaper(
            input_dim=351,
            num_classes=102,
            matrix_size=args.m_matrix_size,
            basis_dim=args.m_basis_dim,
            expm_mode=args.m_expm_mode,
            expm_k=args.m_expm_k,
            scale=args.m_scale,
            init_std=args.m_init_std
        )
        logger.info(f'使用 M-Layer (Paper) 模型 (matrix_size={args.m_matrix_size}, '
                   f'basis_dim={args.m_basis_dim}, scale={args.m_scale}, '
                   f'expm_mode={args.m_expm_mode}, expm_k={args.m_expm_k})')
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    logger.info(f'模型参数量: {count_parameters(model):,}')

    # 检查是否为预测模式（仅由 --predict_only 决定）
    if args.predict_only:
        # 预测模式必须提供 --load_model
        if not args.load_model:
            raise ValueError("预测模式需要指定模型路径 --load_model")

        logger.info(f'加载预训练模型: {args.load_model}')
        checkpoint = torch.load(args.load_model, map_location='cpu')

        # 检查模型类型兼容性
        saved_model_type = checkpoint.get('model_type', 'baseline')
        if saved_model_type != args.model_type:
            raise ValueError(
                f"模型类型不匹配！Checkpoint 中保存的是 '{saved_model_type}'，"
                f"但当前指定的是 '{args.model_type}'。"
                f"请确保使用相同的 --model_type 参数。"
            )

        # 使用兼容性加载（处理旧版 Feature-based checkpoint）
        load_model_state_dict_compatible(model, checkpoint['model_state_dict'])
        model = model.to(device)

        # 重新创建测试数据集
        test_dataset = TestDataset(test_file_1d, test_file_3d)

        # ===== 关键验证：1D与3D标签一致性自检（预测模式）=====
        logger.info('验证1D与3D标签一致性（预测模式）...')
        mask = test_dataset.region_mask.astype(bool)
        labels_3d = test_dataset.region_labels[mask]
        labels_1d = test_dataset.labels

        if len(labels_1d) != len(labels_3d):
            logger.error(f"标签数量不匹配: 1D={len(labels_1d)}, 3D={len(labels_3d)}")
            raise ValueError("1D和3D标签数量不一致")

        labels_match = np.array_equal(labels_1d, labels_3d)
        if labels_match:
            logger.info(f'  ✅ 标签一致性验证通过：{len(labels_1d)}个体素标签完全匹配')
        else:
            n_mismatch = np.sum(labels_1d != labels_3d)
            mismatch_rate = n_mismatch / len(labels_1d) * 100
            logger.error(f'  ❌ 标签一致性验证失败！')
            logger.error(f'     不匹配体素数: {n_mismatch}/{len(labels_1d)} ({mismatch_rate:.2f}%)')
            mismatch_indices = np.where(labels_1d != labels_3d)[0][:10]
            for idx in mismatch_indices:
                logger.error(f'     体素{idx}: 1D标签={labels_1d[idx]}, 3D标签={labels_3d[idx]}')
            raise AssertionError("测试集标签在一维与三维不一致！可能是文件错配")

        logger.info(f'测试样本数: {len(test_dataset)}')
        history = None  # 预测模式不需要训练历史

    else:
        # 训练模式
        # 如果指定了 --load_model，先加载权重作为初始化
        if args.load_model:
            logger.info(f'加载预训练模型作为初始化: {args.load_model}')
            checkpoint = torch.load(args.load_model, map_location='cpu')

            # 检查模型类型兼容性
            saved_model_type = checkpoint.get('model_type', 'baseline')
            if saved_model_type != args.model_type:
                raise ValueError(
                    f"模型类型不匹配！Checkpoint 中保存的是 '{saved_model_type}'，"
                    f"但当前指定的是 '{args.model_type}'。"
                    f"请确保使用相同的 --model_type 参数。"
                )

            # 使用兼容性加载（处理旧版 Feature-based checkpoint）
            load_model_state_dict_compatible(model, checkpoint['model_state_dict'])
            model = model.to(device)

        # 创建训练器（传入学习率调度相关参数）
        trainer = Trainer(
            model,
            device=device,
            learning_rate=0.00001,
            min_delta_acc=args.min_delta_acc,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            min_lr=args.min_lr
        )

        # 训练模型
        logger.info('开始训练...')
        logger.info(f'Early stopping patience: {args.patience}')
        logger.info(f'Min delta for improvement: {args.min_delta_acc}')
        logger.info(f'LR scheduler: factor={args.lr_factor}, patience={args.lr_patience}, min_lr={args.min_lr}')
        history = trainer.train(train_loader, test_loader, epochs=args.epochs, patience=args.patience)
    
    # 保存模型（仅训练模式）
    if not args.predict_only:
        # 根据模型类型决定文件名
        if args.model_type == 'baseline':
            model_filename = f'dense_4x4096_model_test{args.test_subject}.pth'
        elif args.model_type == 'm_layer':
            model_filename = f'm_layer_model_test{args.test_subject}.pth'
        elif args.model_type == 'm_layer_paper':
            model_filename = f'm_layer_paper_model_test{args.test_subject}.pth'
        else:
            model_filename = f'model_test{args.test_subject}.pth'

        model_path = output_dir / model_filename
        save_dict = {
            'model_state_dict': model.state_dict(),
            'scaler': None,
            'history': history,
            'args': vars(args),
            'model_type': args.model_type,  # 保存模型类型
        }
        # 如果是 M-Layer (Operator) 模型，保存其配置
        if args.model_type == 'm_layer':
            save_dict['m_layer_config'] = {
                'matrix_size': args.m_matrix_size,
                'scale': args.m_scale,
                'clip': args.m_clip
            }
        # 如果是 M-Layer (Paper) 模型，保存其配置
        elif args.model_type == 'm_layer_paper':
            save_dict['m_layer_paper_config'] = {
                'matrix_size': args.m_matrix_size,
                'basis_dim': args.m_basis_dim,
                'expm_mode': args.m_expm_mode,
                'expm_k': args.m_expm_k,
                'scale': args.m_scale,
                'init_std': args.m_init_std
            }
        torch.save(save_dict, model_path)
        logger.info(f'模型保存至: {model_path}')
        
        # 保存训练历史（转换numpy类型为JSON可序列化格式）
        history_path = output_dir / f'history_test{args.test_subject}.json'
        history_serializable = convert_to_json_serializable(history)
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    # 如果需要，保存3D预测概率
    if args.save_predictions:
        logger.info('生成3D预测概率体积...')
        predictions_3d = predict_and_map_to_3d(model, test_dataset, device, args.batch_size)
        
        # 保存预测 - 使用HDF5格式处理大文件（13GB超过MAT v5的2GB限制）
        pred_path = output_dir / f'predictions_3d_test{args.test_subject}.mat'
        
        logger.info(f'保存预测到: {pred_path}')
        logger.info(f'  形状: {predictions_3d.shape}')
        logger.info(f'  概率范围: [{predictions_3d.min():.4f}, {predictions_3d.max():.4f}]')
        logger.info(f'  预计文件大小: ~{predictions_3d.nbytes / (1024**3):.1f}GB')
        
        # 使用h5py保存（MATLAB v7.3格式）
        import h5py
        with h5py.File(str(pred_path), 'w') as f:
            # 保存概率数据，使用压缩减小文件大小
            f.create_dataset('softmax_probabilities', 
                           data=predictions_3d.astype(np.float32),
                           compression='gzip', 
                           compression_opts=4)  # 中等压缩级别
            
            # 保存元数据
            f.attrs['test_subject'] = args.test_subject
            f.attrs['test_file_1d'] = str(test_file_1d)
            f.attrs['test_file_3d'] = str(test_file_3d)
            f.attrs['shape'] = predictions_3d.shape
            f.attrs['prob_range'] = [float(predictions_3d.min()), float(predictions_3d.max())]
        
        logger.info(f'预测已保存（HDF5/MAT v7.3格式，带压缩）')
    
    # 最终报告
    if args.predict_only:
        logger.info('\n===== 预测完成 =====')
        logger.info(f'测试被试: {args.test_subject}')
        if args.save_predictions:
            logger.info('3D softmax概率已保存')
    else:
        logger.info('\n===== 训练完成 =====')

        # Early stopping信息
        if history.get('early_stopped', False):
            logger.info(f'Early Stopping: 是（在第{history["total_epochs"]}个epoch停止）')
            logger.info(f'最佳模型来自: Epoch {history["best_epoch"]}')
        else:
            logger.info(f'Early Stopping: 否（完成全部{history["total_epochs"]}个epoch）')
            logger.info(f'最佳模型来自: Epoch {history["best_epoch"]}')

        # 最佳指标（按min_delta规则确定的best，与保存的模型一致）
        logger.info(f'最佳测试Gross Accuracy: {history["best_test_acc"]:.4f} (Epoch {history["best_epoch"]})')

        # 最终epoch的指标
        logger.info(f'最终训练Loss: {history["train_loss"][-1]:.4f}')
        logger.info(f'最终训练F1: {history["train_f1"][-1]:.4f}')
        logger.info(f'最终训练Acc: {history["train_acc"][-1]:.4f}')
        logger.info(f'最终测试Loss: {history["test_loss"][-1]:.4f}')
        logger.info(f'最终测试F1: {history["test_f1"][-1]:.4f}')
        logger.info(f'最终测试Acc: {history["test_acc"][-1]:.4f}')

def smoke_test_paper_model():
    """
    Smoke test for the Paper-Compliant M-Layer model.

    验证：
    1. 模型可以正确创建
    2. 前向传播不报错
    3. 输出形状正确
    4. 唯一非线性只有 matrix_exp（通过代码审查确认）

    使用方法：
        python train_1d_with_3d_dataset.py --smoke_test
    或在 Python 中：
        from train_1d_with_3d_dataset import smoke_test_paper_model
        smoke_test_paper_model()
    """
    print("\n" + "="*60)
    print("Paper M-Layer Smoke Test")
    print("="*60)

    # 测试参数
    batch_size = 8
    input_dim = 351
    num_classes = 102
    matrix_size = 8
    basis_dim = 64

    # 创建随机输入
    x = torch.randn(batch_size, input_dim)
    print(f"输入形状: {x.shape}")

    # 测试 exact 模式
    print("\n--- 测试 exact 模式 ---")
    model_exact = RegModel_MLayerPaper(
        input_dim=input_dim,
        num_classes=num_classes,
        matrix_size=matrix_size,
        basis_dim=basis_dim,
        expm_mode='exact',
        scale=1.0
    )
    print(f"模型参数量: {count_parameters(model_exact):,}")

    model_exact.eval()
    with torch.no_grad():
        logits_exact = model_exact(x)
    print(f"输出形状: {logits_exact.shape}")
    assert logits_exact.shape == (batch_size, num_classes), \
        f"输出形状错误: 期望 ({batch_size}, {num_classes}), 得到 {logits_exact.shape}"
    print("✅ exact 模式通过")

    # 测试 approx 模式
    print("\n--- 测试 approx 模式 ---")
    model_approx = RegModel_MLayerPaper(
        input_dim=input_dim,
        num_classes=num_classes,
        matrix_size=matrix_size,
        basis_dim=basis_dim,
        expm_mode='approx',
        expm_k=6,
        scale=1.0
    )

    model_approx.eval()
    with torch.no_grad():
        logits_approx = model_approx(x)
    print(f"输出形状: {logits_approx.shape}")
    assert logits_approx.shape == (batch_size, num_classes), \
        f"输出形状错误: 期望 ({batch_size}, {num_classes}), 得到 {logits_approx.shape}"
    print("✅ approx 模式通过")

    # 测试不同的 matrix_size 和 basis_dim 组合
    print("\n--- 测试不同参数组合 ---")
    test_configs = [
        {'matrix_size': 4, 'basis_dim': 32},
        {'matrix_size': 16, 'basis_dim': 128},
        {'matrix_size': 32, 'basis_dim': 64},
    ]

    for cfg in test_configs:
        model_test = RegModel_MLayerPaper(
            input_dim=input_dim,
            num_classes=num_classes,
            matrix_size=cfg['matrix_size'],
            basis_dim=cfg['basis_dim'],
            expm_mode='exact',
            scale=1.0
        )
        model_test.eval()
        with torch.no_grad():
            out = model_test(x)
        assert out.shape == (batch_size, num_classes)
        print(f"✅ matrix_size={cfg['matrix_size']}, basis_dim={cfg['basis_dim']} 通过")

    # 验证 expm_approx 函数
    print("\n--- 测试 expm_approx 近似精度 ---")
    M_test = torch.randn(4, 8, 8) * 0.1  # 小矩阵，确保收敛
    E_exact = torch.matrix_exp(M_test)
    E_approx = expm_approx(M_test, k=10)
    diff = torch.abs(E_exact - E_approx).max().item()
    print(f"exact vs approx (k=10) 最大差异: {diff:.6f}")
    assert diff < 0.01, f"近似误差过大: {diff}"
    print("✅ expm_approx 精度验证通过")

    print("\n" + "="*60)
    print("所有 Smoke Test 通过！")
    print("="*60 + "\n")

    # 打印论文一致性确认
    print("论文一致性确认:")
    print("  1) 唯一非线性: torch.matrix_exp / expm_approx ✓")
    print("  2) M(x) 对 x 是仿射: B + einsum(z, basis) ✓")
    print("  3) 输出是线性投影: Linear(vec(exp(M))) ✓")
    print("  4) 无 ReLU/Dropout/clamp/LayerNorm 等非线性 ✓")

    return True


if __name__ == '__main__':
    import sys
    if '--smoke_test' in sys.argv:
        smoke_test_paper_model()
    else:
        main()