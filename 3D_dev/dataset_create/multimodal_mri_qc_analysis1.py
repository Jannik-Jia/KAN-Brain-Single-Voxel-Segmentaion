#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态MRI数据的跨模态一致性与配准质量定量分析 (v1.2.0)

本脚本将Jupyter Notebook中的分析流程自动化，用于批量处理文件夹中所有的.mat文件。
每个.mat文件被视为一个独立的病人数据，其分析结果将保存在以该文件名命名的单独子文件夹中。

分析任务包括：
1. 模态间相似性矩阵 (LNCC, NGF, MIND-SSD)
2. 边缘结构一致性 (ASSD, HD95, Edge IOU)
3. ROI区域一致性 (基于FreeSurfer标签的信号分析)
4. 降维可视化 (PCA聚类分析)
5. QC评分聚合 (综合评分与PASS/WARN/FAIL判断)
6. 面板级可视化导出 (棋盘格、边缘叠加、透明度融合GIF)
   - 三个方向 (axial, coronal, sagittal)
   - 三种视图 (checkerboard, edge_overlay, fading_gif)
   - 像素级拼接，无重采样，输出无损PNG/GIF
"""

# ====================
# 导入库和环境配置 (Cell 1 & 14)
# ====================
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
from tqdm.auto import tqdm
import sys

# 图像处理
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import filters, feature, morphology
from skimage.metrics import structural_similarity

# 机器学习
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 面板可视化（像素级处理）
from PIL import Image, ImageDraw, ImageFont
import imageio

# 过滤警告
warnings.filterwarnings('ignore')

# 设置绘图样式
# Use Latin fonts only to ensure pure-English rendering
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')
# 禁用matplotlib的交互式显示，以便在脚本中运行
plt.ioff()

# ====================
# 全局配置参数 (Cell 2 & 14)
# ====================

# --- 用户配置区域 ---
# 数据路径: 包含所有.mat文件的目录
DATA_DIR = "/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_minimal"  # 包含所有mat文件的目录
# 输出目录: 将在此目录下创建每个病人的子文件夹
BASE_OUTPUT_DIR = Path(DATA_DIR).parent / "qc_analysis_results_batch"
# FreeSurfer标签映射文件 (请确保此路径在您的系统上是正确的)
LABEL_MAPPING_FILE = "/home/jovyan/gpu_space/workspace_jiayi/KAN-git/KAN-Brain-Single-Voxel-Segmentaion/3D_dev/training/downsampling/Freesurfer_LUT_alex_labels_jiayi.xlsx"
# --- 结束配置 ---

# 全局常量
SPACING = (0.65, 0.65, 0.65)  # mm, MPRAGE空间 (German 2021)
RANDOM_SEED = 42
VERSION = "v1.2.0"

# 模态定义
SELECTED_MODALITIES = [
    0, 1, 4, 7, 14,
    20, 50, 100, 160, 220,
    225, 226, 227,
    235, 240, 250, 260, 275,
    229, 284,
    341
]
MODALITY_FAMILIES = {
    'QTI': list(range(0, 15)),
    'DWI': list(range(15, 225)),
    'CEST': list(range(225, 341)),
    'MPRAGE': [341],
    'QSM': list(range(342, 351))
}
MODALITY_NAMES = {
    0: 'QTI_μFA', 1: 'QTI_MD', 4: 'QTI_Kμ', 7: 'QTI_FA', 14: 'QTI_RD',
    20: 'DWI_b20', 50: 'DWI_b50', 100: 'DWI_b100', 160: 'DWI_b160', 220: 'DWI_b220',
    225: 'CEST_Water', 226: 'CEST_NOE', 227: 'CEST_MT',
    235: 'Z_-4ppm', 240: 'Z_-2ppm', 250: 'Z_0ppm', 260: 'Z_2ppm', 275: 'Z_5ppm',
    229: 'M0_B1_0.7', 284: 'M0_B1_1.0',
    341: 'MPRAGE',
}
for idx in SELECTED_MODALITIES:
    if idx not in MODALITY_NAMES:
        for family, indices in MODALITY_FAMILIES.items():
            if idx in indices:
                MODALITY_NAMES[idx] = f"{family}_{idx}"
                break

# ROI定义
KEY_ROIS = {
    'Thalamus': 5,
    'Caudate': 6,
    'Putamen': 7,
    'Pallidum': 8,
    'Hippocampus': 12,
    'Amygdala': 13,
    'Cerebellum_Cortex': 4
}

# QC阈值 (此处保留，但不再用于绘图)
QC_THRESHOLDS = {
    'lncc_min': 0.3,
    'ngf_min': 0.4,
    'assd_max': 2.5,
    'hd95_max': 10.0,
    'edge_iou_min': 0.3,
    'mad_multiplier': 3.0
}
REFERENCE_MODALITY = 341  # MPRAGE

# ===== 面板可视化配置 =====
PANEL_ENABLED            = True       # 总开关
SLICE_STRIDE             = 15         # 每隔多少层出一张面板
MIN_BRAIN_RATIO          = 0.05       # 该slice脑实质占比低于阈值则跳过
TILE_PX                  = 32         # 棋盘格单块像素边长（checkerboard）
EDGE_SIGMA               = 1.0        # Canny平滑sigma
SAVE_REF_EDGE_ON_MOD     = True       # 边缘叠加：绘制MPRAGE边缘于模态底图
SAVE_DUAL_EDGES          = False      # 同时绘制模态自身边缘（洋红）
FADING_FRAMES_K          = 20         # 融合GIF帧数（alpha: 1→0→1）
GIF_FPS                  = 20         # 20fps配合K=20 -> 每秒一个往返循环
TILES_PER_ROW            = 4          # 面板中每行放多少个子图tile
TILE_PAD_PX              = 2          # 子图之间的像素级间隔
ANNOTATION_BAND_PX       = 22         # 每个tile顶部预留文字带像素高
PANEL_BG_VAL             = 0          # 面板背景灰度(0~255）
PNG_COMPRESS_LEVEL       = 0          # PNG压缩等级(0~9)，0避免耗时与重编码

# ====================
# 辅助函数 (Cells 3, 5, 8, 14)
# ====================

def load_minimal_3d_data(mat_path):
    """
    加载3D minimal数据 (来自 Cell 3)
    """
    print(f"\n📂 加载数据文件: {Path(mat_path).name}")
    data_dict = {}
    with h5py.File(mat_path, 'r') as f:
        data = f['data'][:]
        data = np.moveaxis(data, 0, -1)
        data_dict['data'] = data
        data_dict['region_mask'] = f['region_mask'][:]
        data_dict['region_labels'] = f['region_labels'][:]
    
    assert data_dict['data'].shape == (384, 336, 256, 351), "数据维度错误"
    n_roi_voxels = int(np.sum(data_dict['region_mask']))
    print(f"   - ROI体素数: {n_roi_voxels:,}")


    print(f"✅ 数据加载成功")
    print(f"   - 数据形状: {data_dict['data'].shape}")
    return data_dict

def extract_modality(data_4d, modality_idx, mask=None):
    """
    提取单个模态的3D图像 (来自 Cell 3)
    """
    img = data_4d[:, :, :, modality_idx].copy()
    if mask is not None:
        img = img * mask
    return img

# --- 任务1 辅助函数 (Cell 5) ---
def compute_lncc(img1, img2, mask, window_size=7):
    window = np.ones((window_size, window_size, window_size)) / (window_size ** 3)
    mean1 = ndimage.convolve(img1 * mask, window, mode='constant')
    mean2 = ndimage.convolve(img2 * mask, window, mode='constant')
    var1 = ndimage.convolve((img1 * mask - mean1) ** 2, window, mode='constant')
    var2 = ndimage.convolve((img2 * mask - mean2) ** 2, window, mode='constant')
    cov = ndimage.convolve((img1 * mask - mean1) * (img2 * mask - mean2), window, mode='constant')
    denominator = np.sqrt(var1 * var2) + 1e-10
    lncc = cov / denominator
    return np.mean(lncc[mask])

def compute_ngf(img1, img2, mask, epsilon=1e-5):
    grad1 = np.gradient(img1)
    grad2 = np.gradient(img2)
    mag1 = np.sqrt(sum(g**2 for g in grad1)) + epsilon
    mag2 = np.sqrt(sum(g**2 for g in grad2)) + epsilon
    norm_grad1 = [g / mag1 for g in grad1]
    norm_grad2 = [g / mag2 for g in grad2]
    dot_product = sum(g1 * g2 for g1, g2 in zip(norm_grad1, norm_grad2))
    return np.mean(dot_product[mask])

def compute_mind_ssd_simplified(img1, img2, mask, patch_size=3):
    coords = np.argwhere(mask)
    n_samples = min(1000, len(coords))
    np.random.seed(RANDOM_SEED) # 保证采样一致性
    sample_indices = np.random.choice(len(coords), n_samples, replace=False)
    sampled_coords = coords[sample_indices]
    ssd_sum = 0.0
    r = patch_size // 2
    for coord in sampled_coords:
        x, y, z = coord
        x1, x2 = max(0, x-r), min(img1.shape[0], x+r+1)
        y1, y2 = max(0, y-r), min(img1.shape[1], y+r+1)
        z1, z2 = max(0, z-r), min(img1.shape[2], z+r+1)
        patch1 = img1[x1:x2, y1:y2, z1:z2]
        patch2 = img2[x1:x2, y1:y2, z1:z2]
        ssd_sum += np.sum((patch1 - patch2) ** 2)
    return ssd_sum / n_samples if n_samples > 0 else 0.0

# --- 任务1 辅助函数 (Cell 8) ---
def detect_outliers_mad(scores, multiplier=3.0):
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    threshold = median - multiplier * mad
    outliers = scores < threshold
    return outliers, threshold

# --- 任务2 辅助函数 (Cell 14) ---
def extract_edges_canny3d_adaptive(img, mask, sigma=1.0, axis=2, use_otsu=True):
    img = img.astype(np.float32)
    if np.any(mask):
        v1, v99 = np.percentile(img[mask], [1, 99])
    else:
        v1, v99 = img.min(), img.max()
    img_norm = np.clip((img - v1) / (v99 - v1 + 1e-10), 0, 1)
    edges = np.zeros_like(mask, dtype=bool)
    n_slices = img.shape[axis]
    for k in range(n_slices):
        slicer = [slice(None)] * 3
        slicer[axis] = k
        slicer = tuple(slicer)
        slice_img = img_norm[slicer]
        slice_mask = mask[slicer]
        if not np.any(slice_mask):
            continue
        try:
            grad = filters.sobel(slice_img)
            grad_masked = grad[slice_mask]
            if len(grad_masked) > 0:
                try:
                    threshold = filters.threshold_otsu(grad_masked)
                    low_t, high_t = threshold * 0.5, threshold * 1.0
                except:
                    low_t, high_t = 0.1, 0.2
            else:
                low_t, high_t = 0.1, 0.2
            edge_2d = feature.canny(slice_img, sigma=sigma, low_threshold=low_t, high_threshold=high_t)
            edges[slicer] = edge_2d & slice_mask
        except Exception as e:
            continue
    return edges

def extract_edges_multiplane(img, mask, sigma=1.0, use_otsu=True):
    edges_axial = extract_edges_canny3d_adaptive(img, mask, sigma, axis=2, use_otsu=use_otsu)
    edges_coronal = extract_edges_canny3d_adaptive(img, mask, sigma, axis=1, use_otsu=use_otsu)
    edges_sagittal = extract_edges_canny3d_adaptive(img, mask, sigma, axis=0, use_otsu=use_otsu)
    return edges_axial | edges_coronal | edges_sagittal

def compute_assd_fast(edges1, edges2, spacing=SPACING):
    from scipy.ndimage import distance_transform_edt
    if not np.any(edges1) or not np.any(edges2):
        return np.nan
    dist1_to_2 = distance_transform_edt(~edges2, sampling=spacing)
    dist2_to_1 = distance_transform_edt(~edges1, sampling=spacing)
    distances_1to2 = dist1_to_2[edges1]
    distances_2to1 = dist2_to_1[edges2]
    return (np.mean(distances_1to2) + np.mean(distances_2to1)) / 2.0

def compute_hd95_fast(edges1, edges2, spacing=SPACING):
    from scipy.ndimage import distance_transform_edt
    if not np.any(edges1) or not np.any(edges2):
        return np.nan
    dist1_to_2 = distance_transform_edt(~edges2, sampling=spacing)
    dist2_to_1 = distance_transform_edt(~edges1, sampling=spacing)
    distances_1to2 = dist1_to_2[edges1]
    distances_2to1 = dist2_to_1[edges2]
    hd95_1to2 = np.percentile(distances_1to2, 95)
    hd95_2to1 = np.percentile(distances_2to1, 95)
    return max(hd95_1to2, hd95_2to1)

def compute_edge_iou(edges1, edges2):
    intersection = np.logical_and(edges1, edges2).sum()
    union = np.logical_or(edges1, edges2).sum()
    return intersection / union if union > 0 else 0.0

def compute_gradient_correlation(img1, img2, mask, spacing=SPACING):
    grad1 = np.gradient(img1, *spacing)
    grad2 = np.gradient(img2, *spacing)
    mag1 = np.sqrt(sum(g**2 for g in grad1))
    mag2 = np.sqrt(sum(g**2 for g in grad2))
    significant_mask = mask & (mag1 > np.percentile(mag1[mask], 25)) & (mag2 > np.percentile(mag2[mask], 25))
    if not np.any(significant_mask):
        return 0.0
    grad1_norm = np.array([g[significant_mask] / (mag1[significant_mask] + 1e-10) for g in grad1])
    grad2_norm = np.array([g[significant_mask] / (mag2[significant_mask] + 1e-10) for g in grad2])
    dot_product = np.sum(grad1_norm * grad2_norm, axis=0)
    return np.mean(dot_product)

# --- 任务3 辅助函数 (Cell 14) ---
def compute_roi_statistics(img, labels, roi_id):
    roi_mask = (labels == roi_id)
    if not np.any(roi_mask):
        return {'mean': np.nan, 'std': np.nan, 'n_voxels': 0}
    roi_values = img[roi_mask]
    return {
        'mean': np.mean(roi_values),
        'std': np.std(roi_values),
        'n_voxels': len(roi_values)
    }

def compute_roi_contrast(img, labels, roi_id, neighbor_ids=None, brain_mask=None):
    roi_mask = (labels == roi_id)
    if not np.any(roi_mask):
        return np.nan
    if neighbor_ids is None:
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(roi_mask, iterations=3)
        neighbor_mask = dilated & ~roi_mask
        if brain_mask is not None:
            neighbor_mask &= brain_mask
    else:
        neighbor_mask = np.isin(labels, neighbor_ids)
        if brain_mask is not None:
            neighbor_mask &= brain_mask
    if not np.any(neighbor_mask):
        return np.nan
    roi_mean = np.mean(img[roi_mask])
    neighbor_mean = np.mean(img[neighbor_mask])
    contrast = abs(roi_mean - neighbor_mean) / (roi_mean + neighbor_mean + 1e-10)
    return contrast

# --- 任务5 辅助函数 (Cell 14) ---
def get_modality_family(mod_idx):
    for family, indices in MODALITY_FAMILIES.items():
        if mod_idx in indices:
            return family
    return 'Unknown'


# ===== 面板可视化：工具函数 =====

def get_slice_2d(vol3d, axis, idx):
    """
    从3D体数据中提取2D切片，保持原像素不转置。
    axis=0 -> sagittal, axis=1 -> coronal, axis=2 -> axial
    """
    if axis == 0:
        return vol3d[idx, :, :].copy()
    elif axis == 1:
        return vol3d[:, idx, :].copy()
    elif axis == 2:
        return vol3d[:, :, idx].copy()
    else:
        raise ValueError(f"Invalid axis: {axis}")


def normalize_slice_percentile(slc, mask2d):
    """
    使用掩膜内的1-99百分位数归一化切片到[0,1]，掩膜外置0。
    返回float32，范围[0,1]。
    """
    slc_norm = slc.astype(np.float32).copy()
    if not np.any(mask2d):
        return slc_norm * 0

    masked_vals = slc_norm[mask2d]
    if len(masked_vals) == 0:
        return slc_norm * 0

    p1, p99 = np.percentile(masked_vals, [1, 99])
    slc_norm = np.clip((slc_norm - p1) / (p99 - p1 + 1e-10), 0, 1)
    slc_norm[~mask2d] = 0
    return slc_norm


def make_text_band(width, height, text, bg_val=128):
    """
    生成一个文字注记带（灰度），使用PIL绘制文字。
    返回 uint8 数组，shape=(height, width)。
    """
    img = Image.new('L', (width, height), color=bg_val)
    draw = ImageDraw.Draw(img)

    # 尝试加载字体，失败则用默认
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=14)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except:
            font = ImageFont.load_default()

    # 绘制文字（居中）
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # 旧版PIL
        text_width, text_height = draw.textsize(text, font=font)

    x = max(0, (width - text_width) // 2)
    y = max(0, (height - text_height) // 2)
    draw.text((x, y), text, fill=255, font=font)

    return np.array(img, dtype=np.uint8)


def build_checkerboard_tile(ref2d_norm, mod2d_norm, tile_px, annotation_text="", annotation_height=ANNOTATION_BAND_PX):
    """
    生成棋盘格 tile（灰度 uint8）。
    ref2d_norm, mod2d_norm: [0,1] float32
    返回 (H+annotation_height) × W 的 uint8 数组。
    """
    h, w = ref2d_norm.shape

    # 向量化生成棋盘格掩膜
    i_indices = np.arange(h)[:, np.newaxis]  # (h, 1)
    j_indices = np.arange(w)[np.newaxis, :]  # (1, w)
    checker_mask = ((i_indices // tile_px) + (j_indices // tile_px)) % 2 == 0

    # 应用掩膜
    checker = np.where(checker_mask, ref2d_norm, mod2d_norm)
    checker_uint8 = (checker * 255).astype(np.uint8)

    # 添加注记带
    if annotation_height > 0 and annotation_text:
        band = make_text_band(w, annotation_height, annotation_text, bg_val=128)
        checker_uint8 = np.vstack([band, checker_uint8])

    return checker_uint8


def canny_edges_2d(img_norm, mask2d, sigma=EDGE_SIGMA):
    """
    在2D归一化图像上提取Canny边缘。
    返回布尔掩膜。
    """
    if not np.any(mask2d):
        return np.zeros_like(mask2d, dtype=bool)

    try:
        grad = filters.sobel(img_norm)
        grad_masked = grad[mask2d]

        if len(grad_masked) > 0:
            try:
                threshold = filters.threshold_otsu(grad_masked)
                low_t, high_t = threshold * 0.5, threshold * 1.0
            except:
                low_t, high_t = 0.1, 0.2
        else:
            low_t, high_t = 0.1, 0.2

        edges = feature.canny(img_norm, sigma=sigma, low_threshold=low_t, high_threshold=high_t)
        edges = edges & mask2d
        return edges
    except Exception as e:
        return np.zeros_like(mask2d, dtype=bool)


def build_edge_overlay_tile(ref2d_norm, mod2d_norm, mask2d, draw_mod_edge=False,
                           annotation_text="", annotation_height=ANNOTATION_BAND_PX):
    """
    生成边缘叠加 tile（RGB uint8）。
    底图：mod2d_norm 灰度转RGB。
    MPRAGE边缘：青色(0,255,255)。
    若 draw_mod_edge=True，模态自身边缘：洋红(255,0,255)。
    返回 (H+annotation_height) × W × 3 的 uint8 数组。
    """
    h, w = mod2d_norm.shape

    # 底图：模态灰度 -> RGB
    base_gray = (mod2d_norm * 255).astype(np.uint8)
    base_rgb = np.stack([base_gray, base_gray, base_gray], axis=2)

    # MPRAGE边缘
    ref_edges = canny_edges_2d(ref2d_norm, mask2d)
    if SAVE_REF_EDGE_ON_MOD:
        base_rgb[ref_edges] = [0, 255, 255]  # 青色

    # 模态自身边缘（可选）
    if draw_mod_edge and SAVE_DUAL_EDGES:
        mod_edges = canny_edges_2d(mod2d_norm, mask2d)
        base_rgb[mod_edges] = [255, 0, 255]  # 洋红

    # 添加注记带
    if annotation_height > 0 and annotation_text:
        band = make_text_band(w, annotation_height, annotation_text, bg_val=128)
        band_rgb = np.stack([band, band, band], axis=2)
        base_rgb = np.vstack([band_rgb, base_rgb])

    return base_rgb


def build_fading_frames_for_tile(ref2d_norm, mod2d_norm, K, annotation_text="", annotation_height=ANNOTATION_BAND_PX):
    """
    生成透明度融合的帧序列（灰度）。
    alpha: 1.0 → 0.0 → 1.0（K帧），包含端点，循环平滑。
    返回 list of uint8 arrays，每帧 shape=(H+annotation_height, W)。
    """
    frames = []

    # 生成alpha序列：1→0→1，包含端点
    K_down = K // 2 + 1  # 包含 1.0 和 0.0
    alphas = np.concatenate([
        np.linspace(1.0, 0.0, K_down, endpoint=True),
        np.linspace(0.0, 1.0, K - K_down + 1, endpoint=True)[1:]  # 跳过重复的0.0
    ])

    for alpha in alphas:
        blended = alpha * ref2d_norm + (1 - alpha) * mod2d_norm
        blended_uint8 = (blended * 255).astype(np.uint8)

        # 添加注记带
        if annotation_height > 0 and annotation_text:
            band = make_text_band(blended_uint8.shape[1], annotation_height, annotation_text, bg_val=128)
            blended_uint8 = np.vstack([band, blended_uint8])

        frames.append(blended_uint8)

    return frames


def tile_mosaic(tiles, tiles_per_row, pad_px, bg_val):
    """
    将多个 tile 拼接成一个面板（mosaic）。
    tiles: list of numpy arrays (灰度或RGB)，所有tile应有相同的dtype和通道数。
    返回拼接后的大画布。
    """
    if len(tiles) == 0:
        return np.array([[bg_val]], dtype=np.uint8)

    # 检查是灰度还是RGB
    is_rgb = (tiles[0].ndim == 3)

    # 计算行列数
    n_tiles = len(tiles)
    n_rows = (n_tiles + tiles_per_row - 1) // tiles_per_row
    n_cols = min(tiles_per_row, n_tiles)

    # 假设所有tile大小相同
    tile_h, tile_w = tiles[0].shape[:2]

    # 计算面板总尺寸
    panel_h = n_rows * tile_h + (n_rows - 1) * pad_px
    panel_w = n_cols * tile_w + (n_cols - 1) * pad_px

    if is_rgb:
        panel = np.full((panel_h, panel_w, 3), bg_val, dtype=np.uint8)
    else:
        panel = np.full((panel_h, panel_w), bg_val, dtype=np.uint8)

    # 逐tile复制
    for idx, tile in enumerate(tiles):
        row_idx = idx // tiles_per_row
        col_idx = idx % tiles_per_row

        y_start = row_idx * (tile_h + pad_px)
        x_start = col_idx * (tile_w + pad_px)

        panel[y_start:y_start+tile_h, x_start:x_start+tile_w] = tile

    return panel


def save_png_uint8(img, path, compress_level=PNG_COMPRESS_LEVEL):
    """
    保存 uint8 图像为无损PNG。
    img: numpy array (H, W) 或 (H, W, 3)。
    """
    pil_img = Image.fromarray(img)
    pil_img.save(path, format='PNG', compress_level=compress_level)


def save_gif_uint8(frames, path, fps=GIF_FPS):
    """
    保存帧序列为GIF（无限循环）。
    frames: list of numpy arrays (uint8, 灰度或RGB)。
    """
    # 转换为PIL图像列表
    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)
    pil_frames[0].save(path, format='GIF', append_images=pil_frames[1:],
                       save_all=True, duration=duration_ms, loop=0)


# ====================
# 核心处理函数
# ====================

def process_patient(mat_file_path, base_output_dir):
    """
    对单个病人（.mat文件）执行完整的QC分析流程。
    """
    try:
        patient_name = mat_file_path.stem
        patient_output_dir = base_output_dir / patient_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🚀 开始处理病人: {patient_name}")
        print(f"   输出将保存至: {patient_output_dir}")
        print(f"{'='*80}")

        # --- 数据加载 (来自 Cell 4) ---
        mri_data = load_minimal_3d_data(mat_file_path)
        data_4d = mri_data['data']
        brain_mask = mri_data['region_mask'].astype(bool)
        region_labels = mri_data['region_labels']

        # --- 任务1: 模态间相似性矩阵 (来自 Cell 6) ---
        print("\n📌 任务1: 计算模态间相似性矩阵...")
        n_modalities = len(SELECTED_MODALITIES)
        lncc_matrix = np.zeros((n_modalities, n_modalities))
        ngf_matrix = np.zeros((n_modalities, n_modalities))
        mind_matrix = np.zeros((n_modalities, n_modalities))

        # 使用tqdm显示每个病人的内部进度
        for i in tqdm(range(n_modalities), desc="任务1 (相似性)", leave=False):
            idx_i = SELECTED_MODALITIES[i]
            img_i = extract_modality(data_4d, idx_i, brain_mask)
            for j in range(i, n_modalities):
                idx_j = SELECTED_MODALITIES[j]
                img_j = extract_modality(data_4d, idx_j, brain_mask)
                if i == j:
                    lncc_matrix[i, j] = 1.0
                    ngf_matrix[i, j] = 1.0
                    mind_matrix[i, j] = 0.0
                else:
                    lncc = compute_lncc(img_i, img_j, brain_mask)
                    ngf = compute_ngf(img_i, img_j, brain_mask)
                    mind = compute_mind_ssd_simplified(img_i, img_j, brain_mask)
                    
                    lncc_matrix[i, j] = lncc_matrix[j, i] = lncc
                    ngf_matrix[i, j] = ngf_matrix[j, i] = ngf
                    mind_matrix[i, j] = mind_matrix[j, i] = mind
        
        print("✅ 任务1 计算完成")

        # --- 任务1 可视化 (来自 Cell 10) ---
        modality_labels = [MODALITY_NAMES.get(idx, f"Ch{idx}") for idx in SELECTED_MODALITIES]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        sns.heatmap(lncc_matrix, ax=axes[0], cmap='RdYlGn', center=0.5, xticklabels=modality_labels, yticklabels=modality_labels, cbar_kws={'label': 'LNCC'}, vmin=-1, vmax=1)
        axes[0].set_title('LNCC Similarity Matrix', fontsize=14, fontweight='bold')
        sns.heatmap(ngf_matrix, ax=axes[1], cmap='RdYlGn', center=0.5, xticklabels=modality_labels, yticklabels=modality_labels, cbar_kws={'label': 'NGF'}, vmin=-1, vmax=1)
        axes[1].set_title('NGF Similarity Matrix', fontsize=14, fontweight='bold')
        sns.heatmap(mind_matrix, ax=axes[2], cmap='RdYlGn_r', xticklabels=modality_labels, yticklabels=modality_labels, cbar_kws={'label': 'MIND-SSD'})
        axes[2].set_title('MIND-SSD Similarity Matrix\n(smaller = more similar)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = patient_output_dir / 'task1_similarity_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 图表已保存: {save_path}")
        plt.close(fig) # 关闭图形，释放内存

        # --- 任务1 异常检测 (来自 Cell 8) ---
        avg_lncc = np.mean(lncc_matrix, axis=1)
        avg_ngf = np.mean(ngf_matrix, axis=1)
        avg_mind = np.mean(mind_matrix, axis=1)
        
        outliers_lncc, _ = detect_outliers_mad(avg_lncc, QC_THRESHOLDS['mad_multiplier'])
        outliers_ngf, _ = detect_outliers_mad(avg_ngf, QC_THRESHOLDS['mad_multiplier'])
        outliers_mind, _ = detect_outliers_mad(-avg_mind, QC_THRESHOLDS['mad_multiplier'])
        
        # --- 任务2: 边缘结构一致性 (来自 Cell 14) ---
        print("\n📌 任务2: 边缘结构一致性评估...")
        ref_img = extract_modality(data_4d, REFERENCE_MODALITY, brain_mask)
        ref_edges = extract_edges_multiplane(ref_img, brain_mask, sigma=1.0, use_otsu=True)
        print(f"   - 参考边缘点数 (MPRAGE): {np.sum(ref_edges):,}")

        edge_metrics = []
        for i, mod_idx in enumerate(tqdm(SELECTED_MODALITIES, desc="任务2 (边缘)", leave=False)):
            if mod_idx == REFERENCE_MODALITY:
                edge_metrics.append({'modality_idx': mod_idx, 'modality_name': MODALITY_NAMES[mod_idx], 'assd_mm': 0.0, 'hd95_mm': 0.0, 'edge_iou': 1.0, 'grad_corr': 1.0})
                continue
            
            mod_img = extract_modality(data_4d, mod_idx, brain_mask)
            mod_edges = extract_edges_multiplane(mod_img, brain_mask, sigma=1.0, use_otsu=True)
            
            assd = compute_assd_fast(mod_edges, ref_edges, spacing=SPACING)
            hd95 = compute_hd95_fast(mod_edges, ref_edges, spacing=SPACING)
            iou = compute_edge_iou(mod_edges, ref_edges)
            grad_corr = compute_gradient_correlation(mod_img, ref_img, brain_mask, spacing=SPACING)

            edge_metrics.append({'modality_idx': mod_idx, 'modality_name': MODALITY_NAMES[mod_idx], 'assd_mm': assd, 'hd95_mm': hd95, 'edge_iou': iou, 'grad_corr': grad_corr})

        df_edge = pd.DataFrame(edge_metrics)
        print("✅ 任务2 计算完成")

        # --- 任务2 可视化 (来自 Cell 14) ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        df_sorted_assd = df_edge.sort_values('assd_mm', ascending=False)
        # 移除颜色判断和阈值线/图例
        axes[0, 0].barh(df_sorted_assd['modality_name'], df_sorted_assd['assd_mm'], color='steelblue', alpha=0.7)
        # axes[0, 0].axvline(QC_THRESHOLDS['assd_max'], color='red', linestyle='--', label=f"Threshold = {QC_THRESHOLDS['assd_max']} mm")
        axes[0, 0].set_title(f'Average Symmetric Surface Distance (ASSD)', fontweight='bold')
        # axes[0, 0].legend() # 移除图例

        df_sorted_hd = df_edge.sort_values('hd95_mm', ascending=False)
        # 移除颜色判断和阈值线/图例
        axes[0, 1].barh(df_sorted_hd['modality_name'], df_sorted_hd['hd95_mm'], color='steelblue', alpha=0.7)
        # axes[0, 1].axvline(QC_THRESHOLDS['hd95_max'], color='red', linestyle='--', label=f"Threshold = {QC_THRESHOLDS['hd95_max']} mm")
        axes[0, 1].set_title('95% Hausdorff Distance', fontweight='bold')
        # axes[0, 1].legend() # 移除图例

        df_sorted_iou = df_edge.sort_values('edge_iou', ascending=True)
        # 移除颜色判断和阈值线/图例
        axes[1, 0].barh(df_sorted_iou['modality_name'], df_sorted_iou['edge_iou'], color='steelblue', alpha=0.7)
        # axes[1, 0].axvline(QC_THRESHOLDS['edge_iou_min'], color='red', linestyle='--', label=f"Threshold = {QC_THRESHOLDS['edge_iou_min']}")
        axes[1, 0].set_title('Edge Intersection-over-Union', fontweight='bold')
        # axes[1, 0].legend() # 移除图例

        df_sorted_grad = df_edge.sort_values('grad_corr', ascending=True)
        axes[1, 1].barh(df_sorted_grad['modality_name'], df_sorted_grad['grad_corr'], color='steelblue', alpha=0.7)
        axes[1, 1].set_title('Gradient Orientation Similarity', fontweight='bold')
        
        plt.tight_layout()
        save_path = patient_output_dir / 'task2_edge_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 图表已保存: {save_path}")
        plt.close(fig)

        # --- 任务3: ROI区域一致性 (来自 Cell 14) ---
        print("\n📌 任务3: ROI区域一致性分析...")
        roi_analysis = []
        for mod_idx in tqdm(SELECTED_MODALITIES, desc="任务3 (ROI)", leave=False):
            mod_img = extract_modality(data_4d, mod_idx, brain_mask)
            for roi_name, roi_id in KEY_ROIS.items():
                stats = compute_roi_statistics(mod_img, region_labels, roi_id)
                if stats['n_voxels'] > 0:
                    contrast = compute_roi_contrast(mod_img, region_labels, roi_id, brain_mask=brain_mask)
                    roi_analysis.append({
                        'modality': MODALITY_NAMES[mod_idx],
                        'roi': roi_name,
                        'mean_signal': stats['mean'],
                        'std_signal': stats['std'],
                        'contrast': contrast,
                        'n_voxels': stats['n_voxels']
                    })
        df_roi = pd.DataFrame(roi_analysis)
        print("✅ 任务3 计算完成")

        # --- 任务3 可视化 (来自 Cell 14) ---
        pivot_mean = df_roi.pivot(index='modality', columns='roi', values='mean_signal')
        pivot_contrast = df_roi.pivot(index='modality', columns='roi', values='contrast')
        pivot_mean_norm = pivot_mean.div(pivot_mean.max(axis=1), axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        sns.heatmap(pivot_mean_norm, ax=axes[0], cmap='viridis', cbar_kws={'label': 'Normalized Signal Intensity'})
        axes[0].set_title('ROI Signal Intensity Heatmap', fontweight='bold')
        sns.heatmap(pivot_contrast, ax=axes[1], cmap='plasma', cbar_kws={'label': 'Contrast'})
        axes[1].set_title('ROI Contrast Heatmap', fontweight='bold')
        
        plt.tight_layout()
        save_path = patient_output_dir / 'task3_roi_heatmaps.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 图表已保存: {save_path}")
        plt.close(fig)

        # --- 任务4: 降维可视化 (来自 Cell 14) ---
        print("\n📌 任务4: 模态空间降维可视化...")
        modality_features = []
        modality_labels_list = []
        for mod_idx in SELECTED_MODALITIES:
            mod_img = extract_modality(data_4d, mod_idx, brain_mask)
            features = [
                np.mean(mod_img[brain_mask]),
                np.std(mod_img[brain_mask]),
                np.median(mod_img[brain_mask]),
                np.percentile(mod_img[brain_mask], 25),
                np.percentile(mod_img[brain_mask], 75),
                np.min(mod_img[brain_mask]),
                np.max(mod_img[brain_mask])
            ]
            modality_features.append(features)
            modality_labels_list.append(MODALITY_NAMES[mod_idx])
        
        X = np.array(modality_features)
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_scaled)
        
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        clusters = dbscan.fit_predict(X_scaled)
        outliers_pca = (clusters == -1)
        print("✅ 任务4 计算完成")

        # --- 任务4 可视化 (来自 Cell 14) ---
        family_colors = {'QTI': 'red', 'DWI': 'blue', 'CEST': 'green', 'MPRAGE': 'purple', 'QSM': 'orange'}
        colors = []
        for mod_idx in SELECTED_MODALITIES:
            colors.append(family_colors.get(get_modality_family(mod_idx), 'grey'))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=100, alpha=0.7, edgecolors='black')
        for i, label in enumerate(modality_labels_list):
            ax.annotate(label, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.8, xytext=(5, 5), textcoords='offset points')
        if np.any(outliers_pca):
            ax.scatter(X_pca[outliers_pca, 0], X_pca[outliers_pca, 1], s=300, facecolors='none', edgecolors='red', linewidths=2, label='Outlier Modalities')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA Visualization of Modalities', fontsize=14, fontweight='bold')
        legend_elements = [Patch(facecolor=color, label=family) for family, color in family_colors.items()]
        if np.any(outliers_pca):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='red', markersize=10, label='Outlier'))
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        save_path = patient_output_dir / 'task4_pca_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 图表已保存: {save_path}")
        plt.close(fig)


        # --- 任务5: QC评分聚合 (来自 Cell 14) ---
        print("\n📌 任务5: QC评分聚合与判断...")
        qc_results = []
        for i, mod_idx in enumerate(SELECTED_MODALITIES):
            mod_name = MODALITY_NAMES[mod_idx]
            edge_row = df_edge[df_edge['modality_idx'] == mod_idx].iloc[0]
            roi_rows = df_roi[df_roi['modality'] == mod_name]
            
            lncc_score = avg_lncc[i]
            ngf_score = avg_ngf[i]
            mind_score = avg_mind[i]
            assd = edge_row['assd_mm']
            hd95 = edge_row['hd95_mm']
            edge_iou = edge_row['edge_iou']
            grad_corr = edge_row['grad_corr']
            avg_contrast = roi_rows['contrast'].mean() if len(roi_rows) > 0 else np.nan
            
            # (旧的 lncc_norm, ... decision 逻辑已移除)

            qc_results.append({
                'modality_id': mod_idx, 'modality_name': mod_name,
                'modality_family': get_modality_family(mod_idx),
                'mean_lncc': lncc_score, 'mean_ngf': ngf_score, 'mind_ssd': mind_score,
                'assd_mm': assd, 'hd95_mm': hd95, 'edge_iou': edge_iou,
                'grad_angle_corr': grad_corr, 'roi_contrast_avg': avg_contrast,
                # 'qc_score' 和 'decision' 暂时不计算
            })

        df_qc = pd.DataFrame(qc_results)

        # --- 新增：基于分位数的 QC_Score 计算 (B点要求) ---
            
        # 定义向量化的qnorm函数
        def qnorm_vec(x_series, lo, hi):
            # x_series is a pandas Series
            x = x_series.copy()
            # 对于NaN值, 我们将其视为最低分(lo)，以便qnorm返回0
            x = x.fillna(lo) 
            normed = (x - lo) / (hi - lo + 1e-8)
            return np.clip(normed, 0, 1)

        # 计算分位数 (使用 .dropna() 保证鲁棒性)
        assd_q10, assd_q90 = np.nanpercentile(df_qc['assd_mm'].dropna(), [10, 90])
        hd_q10,   hd_q90   = np.nanpercentile(df_qc['hd95_mm'].dropna(), [10, 90])
        lncc_q10, lncc_q90 = np.nanpercentile(df_qc['mean_lncc'].dropna(), [10, 90])
        ngf_q10,  ngf_q90  = np.nanpercentile(df_qc['mean_ngf'].dropna(), [10, 90])
        iou_q10,  iou_q90  = np.nanpercentile(df_qc['edge_iou'].dropna(), [10, 90])
        gc_q10,   gc_q90   = np.nanpercentile(df_qc['grad_angle_corr'].dropna(), [10, 90])

        # 对每行重新计算 qc_score (不使用 QC_THRESHOLDS)
        df_qc['qc_score'] = (
            0.20 * qnorm_vec(df_qc['mean_lncc'], lncc_q10, lncc_q90) +
            0.15 * qnorm_vec(df_qc['mean_ngf'],  ngf_q10,  ngf_q90)  +
            0.20 * (1.0 - qnorm_vec(df_qc['assd_mm'],  assd_q10, assd_q90)) +
            0.15 * (1.0 - qnorm_vec(df_qc['hd95_mm'],  hd_q10,   hd_q90))   +
            0.15 * qnorm_vec(df_qc['edge_iou'],        iou_q10,  iou_q90)   +
            0.15 * qnorm_vec(df_qc['grad_angle_corr'], gc_q10,   gc_q90)
        ) * 100.0
        # --- 结束新增 ---
        
        df_qc = df_qc.sort_values('qc_score', ascending=False)
        df_qc['qc_rank'] = range(1, len(df_qc) + 1)
        
        csv_path = patient_output_dir / 'modality_qc.csv'
        df_qc.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 任务5 计算完成")
        print(f"   💾 QC报告 (CSV) 已保存: {csv_path}")


        # --- 任务5 可视化 (根据要求修改 - 使用分位数标准化雷达图) ---
        
        # 辅助函数：用分位数范围而非硬阈值
        def qnorm(x, low, high):
            if np.isnan(x): return 0.0
            # 避免除以零
            return np.clip((x - low) / (high - low + 1e-8), 0, 1)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        df_qc_sorted = df_qc.sort_values('qc_score', ascending=False)
        # (colors_decision 和 bar_colors 已移除)
        
        axes[0].barh(df_qc_sorted['modality_name'], df_qc_sorted['qc_score'], color='steelblue', alpha=0.7)
        axes[0].set_title(f'Modality Registration Quality Scores - {patient_name}', fontweight='bold')
        # (阈值线和图例已按上一请求移除)

        top_modalities = df_qc.head(6)
        metrics = ['mean_lncc', 'mean_ngf', 'assd_mm', 'hd95_mm', 'edge_iou', 'grad_angle_corr']
        metric_labels = ['LNCC', 'NGF', 'ASSD', 'HD95', 'Edge IoU', 'Grad Corr']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist() + [0]
        
        # --- 新增：计算雷达图标准化所需的分位数 ---
        # 越大越好的指标 (LNCC, NGF, IOU, GradCorr)，使用 (0, 1) 范围或它们自身的(q10, q90)
        lncc_q10, lncc_q90 = np.nanpercentile(df_qc['mean_lncc'], [10, 90])
        ngf_q10, ngf_q90 = np.nanpercentile(df_qc['mean_ngf'], [10, 90])
        iou_q10, iou_q90 = np.nanpercentile(df_qc['edge_iou'], [10, 90])
        grad_q10, grad_q90 = np.nanpercentile(df_qc['grad_angle_corr'], [10, 90])
        
        # ASSD/HD95 (越小越好) 使用10/90分位数
        assd_q10, assd_q90 = np.nanpercentile(df_qc['assd_mm'], [10, 90])
        hd_q10, hd_q90   = np.nanpercentile(df_qc['hd95_mm'], [10, 90])
        # --- 结束新增 ---
        
        ax_radar = plt.subplot(2, 1, 2, projection='polar')
        for _, row in top_modalities.iterrows():
            values = [row[m] for m in metrics]
            values_norm = []
            
            # --- 修改：使用 qnorm 进行标准化 ---
            values_norm.append(qnorm(values[0], lncc_q10, lncc_q90)) # LNCC (越大越好)
            values_norm.append(qnorm(values[1], ngf_q10, ngf_q90))   # NGF (越大越好)
            values_norm.append(1.0 - qnorm(values[2], assd_q10, assd_q90)) # ASSD (越小越好, 反向)
            values_norm.append(1.0 - qnorm(values[3], hd_q10, hd_q90))     # HD95 (越小越好, 反向)
            values_norm.append(qnorm(values[4], iou_q10, iou_q90))   # Edge IoU (越大越好)
            values_norm.append(qnorm(values[5], grad_q10, grad_q90)) # Grad Corr (越大越好)
            # --- 结束修改 ---
                
            values_norm += values_norm[:1] # close the loop
            ax_radar.plot(angles, values_norm, 'o-', linewidth=2, label=row['modality_name'], alpha=0.7)
            ax_radar.fill(angles, values_norm, alpha=0.15)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metric_labels)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Radar Chart of Top 6 Modalities (10/90 Quantile Normalized)', fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        save_path = patient_output_dir / 'task5_qc_scores.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 图表已保存: {save_path}")
        plt.close(fig)



        # --- 最终报告 (来自 Cell 15) ---
        print("\n📄 生成最终分析报告 (JSON)...")
        qc_score_percentiles = {f'p{p}': float(np.percentile(df_qc['qc_score'], p)) for p in [10, 25, 50, 75, 90]}
        assd_percentiles = {f'p{p}': float(np.percentile(df_qc['assd_mm'].dropna(), p)) for p in [10, 50, 90]}
        hd95_percentiles = {f'p{p}': float(np.percentile(df_qc['hd95_mm'].dropna(), p)) for p in [10, 50, 90]}
        
        by_family = {}
        for family in MODALITY_FAMILIES.keys():
            family_df = df_qc[df_qc['modality_family'] == family]
            if len(family_df) > 0:
                by_family[family] = {
                    'n': int(len(family_df)),
                    'mean_qc': float(family_df['qc_score'].mean()),
                    'std_qc': float(family_df['qc_score'].std()),
                    'p25_qc': float(np.percentile(family_df['qc_score'], 25)),
                    'p50_qc': float(np.percentile(family_df['qc_score'], 50)),
                    'p75_qc': float(np.percentile(family_df['qc_score'], 75)),
                }


        report = {
            'subject': mat_file_path.name,
            'analysis_date': datetime.now().isoformat(),
            'version': VERSION,
            'physical_spacing_mm': SPACING,
            'reference_modality': MODALITY_NAMES[REFERENCE_MODALITY],
            'summary': {
                'total_modalities': len(SELECTED_MODALITIES),
                # (pass, warn, fail 已移除)
                'avg_qc_score': float(df_qc['qc_score'].mean()),
                'outliers_detected': int(np.sum(outliers_pca)),
                'qc_score_percentiles': qc_score_percentiles,
                'assd_percentiles': assd_percentiles,
                'hd95_percentiles': hd95_percentiles
            },
            'by_family': by_family,
            'all_modality_scores': df_qc.to_dict('records') # 包含所有模态的详细信息
        }

        json_path = patient_output_dir / 'qc_analysis_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"   💾 JSON报告已保存: {json_path}")


        # --- 任务6: 面板级可视化导出 ---
        if PANEL_ENABLED:
            print("\n📌 任务6: 面板级可视化导出...")

            # 创建输出目录结构
            viz_root = patient_output_dir / "visualizations_panel"
            axis_names = ['axial', 'coronal', 'sagittal']
            view_types = ['checkerboard', 'edge_overlay', 'fading_gif']

            for axis_name in axis_names:
                for view_type in view_types:
                    (viz_root / axis_name / view_type).mkdir(parents=True, exist_ok=True)

            # 提取参考模态（MPRAGE）
            ref_img3d = extract_modality(data_4d, REFERENCE_MODALITY, brain_mask)

            # 获取非参考模态列表
            mod_indices = [m for m in SELECTED_MODALITIES if m != REFERENCE_MODALITY]

            # 记录manifest信息
            manifest = []

            # 遍历三个方向
            for axis_idx, axis_name in enumerate(axis_names):
                n_slices = ref_img3d.shape[axis_idx]
                axis_abbr = {'axial': 'ax', 'coronal': 'co', 'sagittal': 'sa'}[axis_name]

                print(f"   处理 {axis_name} 方向 (共{n_slices}层)...")

                # 按stride采样切片
                for slice_idx in range(0, n_slices, SLICE_STRIDE):
                    # 提取参考切片和掩膜
                    ref2d = get_slice_2d(ref_img3d, axis_idx, slice_idx)
                    mask2d = get_slice_2d(brain_mask.astype(np.float32), axis_idx, slice_idx).astype(bool)

                    # 有效性检查
                    brain_ratio = mask2d.mean()
                    if brain_ratio < MIN_BRAIN_RATIO:
                        print(f"      跳过 {axis_name} slice={slice_idx} (脑实质占比 {brain_ratio:.3f} < {MIN_BRAIN_RATIO})")
                        continue

                    # 归一化参考切片
                    ref2d_norm = normalize_slice_percentile(ref2d, mask2d)

                    # 为每个模态生成tiles
                    checker_tiles = []
                    edge_tiles = []
                    fading_tile_frames = []  # list of list: [mod][frame]

                    for mod_idx in mod_indices:
                        mod_name = MODALITY_NAMES.get(mod_idx, f"Ch{mod_idx}")
                        annotation_text = f"{mod_name} | {axis_name} | slice={slice_idx}"

                        # 提取模态切片
                        mod_img3d = extract_modality(data_4d, mod_idx, brain_mask)
                        mod2d = get_slice_2d(mod_img3d, axis_idx, slice_idx)
                        mod2d_norm = normalize_slice_percentile(mod2d, mask2d)

                        # 生成三种tile
                        checker_tile = build_checkerboard_tile(ref2d_norm, mod2d_norm, TILE_PX,
                                                              annotation_text=annotation_text,
                                                              annotation_height=ANNOTATION_BAND_PX)
                        checker_tiles.append(checker_tile)

                        edge_tile = build_edge_overlay_tile(ref2d_norm, mod2d_norm, mask2d,
                                                           draw_mod_edge=SAVE_DUAL_EDGES,
                                                           annotation_text=annotation_text,
                                                           annotation_height=ANNOTATION_BAND_PX)
                        edge_tiles.append(edge_tile)

                        fading_frames = build_fading_frames_for_tile(ref2d_norm, mod2d_norm, FADING_FRAMES_K,
                                                                    annotation_text=annotation_text,
                                                                    annotation_height=ANNOTATION_BAND_PX)
                        fading_tile_frames.append(fading_frames)

                    # 拼接面板并保存

                    # (1) 棋盘格面板
                    checker_panel = tile_mosaic(checker_tiles, TILES_PER_ROW, TILE_PAD_PX, PANEL_BG_VAL)
                    checker_path = viz_root / axis_name / 'checkerboard' / f"{axis_name}_checker_panel_vsMPRAGE_{axis_abbr}{slice_idx:03d}_tile{TILE_PX}.png"
                    save_png_uint8(checker_panel, checker_path, compress_level=PNG_COMPRESS_LEVEL)

                    # (2) 边缘叠加面板
                    edge_panel = tile_mosaic(edge_tiles, TILES_PER_ROW, TILE_PAD_PX, PANEL_BG_VAL)
                    edge_path = viz_root / axis_name / 'edge_overlay' / f"{axis_name}_edge_panel_MPRAGEonMOD_{axis_abbr}{slice_idx:03d}.png"
                    save_png_uint8(edge_panel, edge_path, compress_level=PNG_COMPRESS_LEVEL)

                    # (3) 融合GIF面板（逐帧拼接）
                    gif_panel_frames = []
                    for frame_idx in range(FADING_FRAMES_K):
                        # 收集所有模态在当前帧的tiles
                        frame_tiles = [fading_tile_frames[mod_i][frame_idx] for mod_i in range(len(mod_indices))]
                        panel_frame = tile_mosaic(frame_tiles, TILES_PER_ROW, TILE_PAD_PX, PANEL_BG_VAL)
                        gif_panel_frames.append(panel_frame)

                    gif_path = viz_root / axis_name / 'fading_gif' / f"{axis_name}_fading_panel_vsMPRAGE_{axis_abbr}{slice_idx:03d}_K{FADING_FRAMES_K}.gif"
                    save_gif_uint8(gif_panel_frames, gif_path, fps=GIF_FPS)

                    # 记录manifest
                    manifest.append({
                        'axis': axis_name,
                        'slice_idx': slice_idx,
                        'brain_ratio': float(brain_ratio),
                        'n_modalities': len(mod_indices),
                        'checker_path': str(checker_path.relative_to(patient_output_dir)),
                        'edge_path': str(edge_path.relative_to(patient_output_dir)),
                        'gif_path': str(gif_path.relative_to(patient_output_dir))
                    })

                    print(f"      ✓ {axis_name} slice={slice_idx} (脑占比={brain_ratio:.2f})")

            # 保存manifest
            manifest_path = viz_root / 'panel_manifest.json'
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            print(f"✅ 任务6 完成，共生成 {len(manifest)} 组面板")
            print(f"   💾 Manifest已保存: {manifest_path}")


        print(f"✅ 完成处理: {patient_name}")

        plt.close('all') # 确保关闭所有图形

    except Exception as e:
        print(f"❌❌❌ 处理 {mat_file_path.name} 时发生错误: {e}")
        # 可以在这里记录更详细的traceback
        import traceback
        traceback.print_exc()
        plt.close('all') # 即使出错也关闭图形


# ====================
# 主执行流程
# ====================

def main():
    """
    主函数，遍历DATA_DIR中的所有.mat文件并处理。
    """
    print(f"======= 开始批量QC分析 (版本 {VERSION}) =======")
    print(f"数据源目录: {DATA_DIR}")
    print(f"基础输出目录: {BASE_OUTPUT_DIR}")
    print("="*40)
    
    # 确保基础输出目录存在
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 查找所有.mat文件
    mat_files = sorted(list(Path(DATA_DIR).glob('*.mat')))
    
    if not mat_files:
        print(f"⚠️ 警告: 在 {DATA_DIR} 中未找到任何 .mat 文件。")
        return

    print(f"🔍 发现 {len(mat_files)} 个 .mat 文件准备处理。")
    
    # 遍历并处理每个文件
    # 使用tqdm为病人列表添加总进度条
    for mat_file_path in tqdm(mat_files, desc="总进度 (病人)"):
        process_patient(mat_file_path, BASE_OUTPUT_DIR)

    print("="*40)
    print("🎉 批量处理全部完成。")
    print(f"所有结果保存在: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()