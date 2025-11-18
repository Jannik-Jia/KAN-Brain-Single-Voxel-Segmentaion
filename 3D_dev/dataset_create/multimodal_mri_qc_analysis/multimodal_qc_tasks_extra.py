#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态MRI QC附加功能脚本 (v1.3.2-extra)
--------------------------------------
在不改变读取路径、数据集键名、保存根路径与高清风格的前提下，
新增/修改如下：
A) ROI 351×351 皮尔逊相关矩阵热图（带“家族/子家族”分块线），高清保存；
B) “一图 351 子图”面板：切片选择遵循原逻辑（首个≥阈值后起点 + 每隔3层），
   且每个 tile 注记显示“模态名（1-based通道号）”，不再出现 ChXXX；
C) Curtain（模态×像素）：保持单幅展示（351×采样像素），无需参考/对比；
D) **新增**：351维度对比图（其他350 vs MPRAGE的透明度切换 GIF 面板），
   切片选择与(B)一致，注记同样使用“模态名”。

读取路径/数据维度/spacing/保存风格与原脚本保持一致。
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 绘图
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['figure.dpi'] = 100  # 屏显DPI；保存统一用 dpi=300

# I/O
try:
    import h5py
except ImportError:
    h5py = None

from PIL import Image, ImageDraw, ImageFont

# -------------------------------
# 1) 复用既有脚本参数/函数；失败则回退
# -------------------------------
FALLBACKS = {}
try:
    import multimodal_mri_qc_analysis1 as base
    DATA_DIR = Path(getattr(base, 'DATA_DIR'))
    BASE_OUTPUT_DIR = Path(getattr(base, 'BASE_OUTPUT_DIR'))
    SPACING = tuple(getattr(base, 'SPACING', (0.65, 0.65, 0.65)))
    REFERENCE_MODALITY = int(getattr(base, 'REFERENCE_MODALITY', 341))  # 0-based MPRAGE
    # 复用加载/切片与归一化工具（若存在）
    load_minimal_3d_data = getattr(base, 'load_minimal_3d_data')
    def _extract_modality(data_4d, modality_idx, mask=None):
        fun = getattr(base, 'extract_modality', None)
        if fun is not None:
            return fun(data_4d, modality_idx, mask)
        img = data_4d[..., modality_idx].copy()
        if mask is not None:
            img = img * mask
        return img
    def _normalize_slice(slc, mask2d):
        fun = getattr(base, 'normalize_slice_percentile', None)
        if fun is not None:
            return fun(slc, mask2d)
        return normalize_slice_percentile(slc, mask2d=None if mask2d is None else mask2d.astype(bool))
except Exception as e:
    DATA_DIR = Path("/home/jovyan/gpu_space/workspace_jiayi/alex_datasets/3D_minimal")
    BASE_OUTPUT_DIR = DATA_DIR.parent / "qc_analysis_results_batch"
    SPACING = (0.65, 0.65, 0.65)
    REFERENCE_MODALITY = 341
    FALLBACKS['reason'] = f"import multimodal_mri_qc_analysis1 failed: {e}"

    def load_minimal_3d_data(mat_path):
        if h5py is None:
            raise RuntimeError("h5py 不可用且无法导入原脚本加载函数。")
        with h5py.File(mat_path, 'r') as f:
            data = f['data'][:]
            if data.shape[0] == 351:  # (351, X, Y, Z) -> (X,Y,Z,351)
                data = np.moveaxis(data, 0, -1)
            data4d = data
            region_mask = f['region_mask'][:].astype(bool)
            region_labels = f['region_labels'][:]
        assert data4d.shape[-1] == 351 and region_mask.shape == region_labels.shape, \
            "数据维度不符合约定：data(...,351), region_mask/region_labels 同形状。"
        return {'data': data4d, 'region_mask': region_mask, 'region_labels': region_labels}

    def _extract_modality(data_4d, modality_idx, mask=None):
        img = data_4d[..., modality_idx].copy()
        if mask is not None:
            img = img * mask
        return img

    def _normalize_slice(slc, mask2d):
        return normalize_slice_percentile(slc, mask2d)

# -------------------------------------------
# 2) 通道“模态名”映射（严格按你给的1-based区间）
#    代码内部使用 0-based；注记展示 1-based
# -------------------------------------------
# 规则按“越具体越先匹配”的顺序（单点/子区间优先，大区间后匹配）
MODALITY_NAME_RULES = [
    # --- 单点/短区间 ---
    ("MPRAGE",              341, 341),  # 342(1-based)
    ("QSM_TE",              342, 346),
    ("TE_avg",              347, 347),
    ("SMWI_on_avg",         348, 348),
    ("SMWI_on_first",       349, 349),
    ("QSM",                 350, 350),
    ("M0",                  229, 229),
    ("M0",                  284, 285),
    ("M0",                  340, 340),
    # --- CEST/Z谱 ---
    ("z spectrum high B1",  286, 339),
    ("z spectrum low B1",   230, 283),
    ("CEST",                225, 228),
    # --- b-tensor 子家族 ---
    ("b_spher",             177, 224),
    ("b_plan",               96, 176),
    ("b_lin",                15,  95),
    # --- QTI ---
    ("QTI",                   0,  14),
    # 备注：“b-tensor 16–225”大区间被子家族覆盖，无需单独规则
]

def modality_name_from_index(idx0: int) -> str:
    """返回最具体的模态名（不含通道号），依据 0-based 索引。"""
    for name, a, b in MODALITY_NAME_RULES:
        if a <= idx0 <= b:
            return name
    return "Unknown"

def modality_display_name(idx0: int) -> str:
    """返回用于注记的模态显示名，例如：'b_lin (16)' —— 括号内为 1-based 通道号。"""
    return f"{modality_name_from_index(idx0)} ({idx0+1})"

# -------------------------------------------
# 3) 家族/子家族分块，用于热图边界
# -------------------------------------------
GROUPS_FOR_BOUNDS = [
    ("QTI",                  0,  14),
    ("b_lin",               15,  95),
    ("b_plan",              96, 176),
    ("b_spher",            177, 224),
    ("CEST",               225, 228),
    ("M0",                 229, 229),
    ("z spectrum low B1",  230, 283),
    ("M0",                 284, 285),
    ("z spectrum high B1", 286, 339),
    ("M0",                 340, 340),
    ("MPRAGE",             341, 341),
    ("QSM_TE",             342, 346),
    ("TE_avg",             347, 347),
    ("SMWI_on_avg",        348, 348),
    ("SMWI_on_first",      349, 349),
    ("QSM",                350, 350),
]

# -------------------------------------------
# 4) 相关矩阵与绘制
# -------------------------------------------
def compute_full_corr_roi(data_4d, roi_mask, max_voxels=200_000, random_seed=42):
    X, Y, Z, C = data_4d.shape
    assert C == 351
    idx_all = np.where(roi_mask.ravel())[0]
    if idx_all.size == 0:
        raise ValueError("ROI 为空，无法计算相关矩阵")
    rng = np.random.default_rng(random_seed)
    if idx_all.size > max_voxels:
        idx = rng.choice(idx_all, size=max_voxels, replace=False)
    else:
        idx = idx_all
    M = data_4d.reshape(-1, C)[idx, :].astype(np.float64)
    M -= M.mean(axis=0, keepdims=True)
    std = M.std(axis=0, ddof=1); std[std == 0] = 1.0
    M /= std[None, :]
    R = (M.T @ M) / (M.shape[0] - 1.0)
    return np.clip(R, -1.0, 1.0)

def _draw_group_boundaries(ax, color='white', lw=0.6):
    for _, a, b in GROUPS_FOR_BOUNDS:
        ax.axhline(b+0.5, color=color, lw=lw)
        ax.axvline(b+0.5, color=color, lw=lw)

def save_corr_heatmap(R, out_png, title="ROI Pearson Corr 351×351", dpi=300):
    plt.figure(figsize=(12,10))
    ax = plt.gca()
    sns.heatmap(R, ax=ax, cmap='viridis', vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False, cbar_kws={'label':'r'})
    _draw_group_boundaries(ax, color='white', lw=0.6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close()

# -------------------------------------------
# 5) 工具：切片/归一化/拼接/注记带
# -------------------------------------------
def get_slice_2d(vol3d, axis, idx):
    if axis == 0: return vol3d[idx, :, :].copy()
    if axis == 1: return vol3d[:, idx, :].copy()
    return vol3d[:, :, idx].copy()

def normalize_slice_percentile(slc, mask2d):
    slc = slc.astype(np.float32)
    if mask2d is None or not np.any(mask2d):
        return slc*0
    p1, p99 = np.percentile(slc[mask2d], [1, 99])
    slc = np.clip((slc - p1)/(p99 - p1 + 1e-8), 0, 1)
    slc[~mask2d] = 0
    return slc

def _make_text_band(width, height, text, bg_val=128):
    img = Image.new('L', (width, height), bg_val)
    draw = ImageDraw.Draw(img)
    # 字体：尽量选择通用字体，失败则退回默认
    for font_path in ["/System/Library/Fonts/Helvetica.ttc", "arial.ttf"]:
        try:
            font = ImageFont.truetype(font_path, size=14)
            break
        except:
            font = None
    if font is None:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except:
        tw, th = draw.textsize(text, font=font)
    x = max(0, (width - tw)//2); y = max(0, (height - th)//2)
    draw.text((x,y), text, fill=255, font=font)
    return np.array(img, dtype=np.uint8)

def build_gray_tile(img2d_norm01, annotate=None, annotation_h=22):
    base = (img2d_norm01 * 255).astype(np.uint8)
    if annotation_h > 0 and annotate:
        band = _make_text_band(base.shape[1], annotation_h, annotate, bg_val=128)
        base = np.vstack([band, base])
    return base

def tile_mosaic(tiles, tiles_per_row, pad_px, bg_val):
    if len(tiles) == 0: return np.array([[bg_val]], dtype=np.uint8)
    is_rgb = (tiles[0].ndim == 3)
    n_tiles = len(tiles)
    n_rows = (n_tiles + tiles_per_row - 1)//tiles_per_row
    n_cols = min(tiles_per_row, n_tiles)
    th, tw = tiles[0].shape[:2]
    H = n_rows*th + (n_rows-1)*pad_px
    W = n_cols*tw + (n_cols-1)*pad_px
    panel = np.full((H, W, 3), bg_val, dtype=np.uint8) if is_rgb else np.full((H, W), bg_val, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, tiles_per_row)
        y = r*(th+pad_px); x = c*(tw+pad_px)
        panel[y:y+th, x:x+tw] = tile
    return panel

def save_png_uint8(img, path, compress_level=0):
    Image.fromarray(img).save(path, format='PNG', compress_level=compress_level)

# -------------------------------------------
# 6) 切片选择规则（与你原代码一致）
# -------------------------------------------
def first_valid_slice(mask3d, axis, min_ratio):
    n = mask3d.shape[axis]
    for k in range(n):
        m2d = get_slice_2d(mask3d, axis, k).astype(bool)
        if m2d.mean() >= min_ratio:
            return k
    return None

# -------------------------------------------
# 7) 功能B：一图351子图（注记=模态名）
# -------------------------------------------
def save_full351_mosaic_for_slice(data_4d, mask3d, axis, slice_idx,
                                  tiles_per_row=27, pad_px=2,
                                  annotation_h=22, out_png=None):
    X, Y, Z, C = data_4d.shape
    assert C == 351
    mask2d = get_slice_2d(mask3d.astype(bool), axis, slice_idx)
    tiles = []
    for ch in range(C):
        img3d = data_4d[..., ch]
        img2d = get_slice_2d(img3d, axis, slice_idx)
        norm = _normalize_slice(img2d, mask2d)
        label = modality_display_name(ch)  # *** 使用模态名 ***
        tile = build_gray_tile(norm, annotate=label, annotation_h=annotation_h)
        tiles.append(tile)
    panel = tile_mosaic(tiles, tiles_per_row=tiles_per_row, pad_px=pad_px, bg_val=0)
    if out_png is not None:
        save_png_uint8(panel, out_png, compress_level=0)
    return panel

# -------------------------------------------
# 8) 功能C：Curtain（351×像素）——无参考/对比
# -------------------------------------------
def save_modalities_curtain(data_4d, roi_mask, out_png, max_voxels=50_000, sort_by=None, dpi=300):
    X, Y, Z, C = data_4d.shape
    assert C == 351
    idx_all = np.where(roi_mask.ravel())[0]
    if idx_all.size == 0:
        raise ValueError("ROI 为空，无法生成 curtain")
    rng = np.random.default_rng(42)
    if idx_all.size > max_voxels:
        cols = rng.choice(idx_all, size=max_voxels, replace=False)
    else:
        cols = idx_all
    if sort_by == 'mprage':
        ref = data_4d.reshape(-1, C)[cols, REFERENCE_MODALITY].astype(np.float32)
        order = np.argsort(ref)
        cols = cols[order]
    curtain_u8 = np.zeros((C, cols.size), dtype=np.uint8)
    flat = data_4d.reshape(-1, C).astype(np.float32)
    for ch in range(C):
        v = flat[cols, ch]
        p1, p99 = np.percentile(v, [1, 99])
        vn = np.clip((v - p1)/(p99 - p1 + 1e-8), 0, 1)
        curtain_u8[ch, :] = (vn*255).astype(np.uint8)
    # 绘制
    plt.figure(figsize=(min(18, 0.00035*curtain_u8.shape[1]+6), 10))
    ax = plt.gca()
    ax.imshow(curtain_u8, cmap='gray', aspect='auto', interpolation='nearest')
    ax.set_xlabel(f'ROI pixels (sampled: {cols.size:,})')
    ax.set_ylabel('Modalities (0-350)')  # 行标签过多不显示具体名称，避免拥挤
    ax.set_title('Curtain: 351 Modalities × ROI Pixels (normalized per modality)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close()
    # 同时输出索引-名称对照，便于查阅
    txt = os.fspath(Path(out_png).with_suffix('.labels.txt'))
    with open(txt, 'w', encoding='utf-8') as f:
        for ch in range(C):
            f.write(f"{ch:03d}\t{modality_display_name(ch)}\n")

# -------------------------------------------
# 9) **新增D**：351维度透明度切换（其他350 vs MPRAGE）
# -------------------------------------------
def build_fading_frames_for_tile(ref2d_norm, mod2d_norm, K, annotation_text="", annotation_height=22):
    frames = []
    K_down = K // 2 + 1
    alphas = np.concatenate([
        np.linspace(1.0, 0.0, K_down, endpoint=True),
        np.linspace(0.0, 1.0, K - K_down + 1, endpoint=True)[1:]
    ])
    for alpha in alphas:
        blended = alpha * ref2d_norm + (1 - alpha) * mod2d_norm
        blended_uint8 = (blended * 255).astype(np.uint8)
        if annotation_height > 0 and annotation_text:
            band = _make_text_band(blended_uint8.shape[1], annotation_height, annotation_text, bg_val=128)
            blended_uint8 = np.vstack([band, blended_uint8])
        frames.append(blended_uint8)
    return frames

def save_gif_uint8(frames, path, fps=20):
    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)
    pil_frames[0].save(path, format='GIF', append_images=pil_frames[1:],
                       save_all=True, duration=duration_ms, loop=0, optimize=False)

def save_fading_panel_all_vs_mprage_for_slice(
        data_4d, mask3d, axis, slice_idx,
        tiles_per_row=27, pad_px=2, annotation_h=22,
        K=20, fps=20, out_gif=None):
    """
    为指定切片生成包含“全部其他350模态 vs MPRAGE”的透明度切换 GIF 面板。
    - 每个tile：顶部注记=模态名（1-based通道号）
    - 面板：像素级拼接，不缩放
    """
    C = data_4d.shape[-1]
    assert C == 351
    mask2d = get_slice_2d(mask3d.astype(bool), axis, slice_idx)
    # 参考（MPRAGE）
    ref2d = get_slice_2d(data_4d[..., REFERENCE_MODALITY], axis, slice_idx)
    ref2d_norm = _normalize_slice(ref2d, mask2d)
    # 预生成所有模态的K帧tile（避免重复归一化）
    tile_frames_per_mod = []  # list(mod) of list(K) of uint8(HxW)
    mod_indices = [ch for ch in range(C) if ch != REFERENCE_MODALITY]
    for ch in mod_indices:
        mod2d = get_slice_2d(data_4d[..., ch], axis, slice_idx)
        mod2d_norm = _normalize_slice(mod2d, mask2d)
        label = modality_display_name(ch)
        frames = build_fading_frames_for_tile(ref2d_norm, mod2d_norm, K,
                                              annotation_text=label,
                                              annotation_height=annotation_h)
        tile_frames_per_mod.append(frames)
    # 逐帧拼接为面板
    panel_frames = []
    for k in range(K):
        frame_tiles = [tile_frames_per_mod[i][k] for i in range(len(mod_indices))]
        panel = tile_mosaic(frame_tiles, tiles_per_row=tiles_per_row, pad_px=pad_px, bg_val=0)
        panel_frames.append(panel)
    # 保存GIF
    if out_gif is not None:
        save_gif_uint8(panel_frames, out_gif, fps=fps)
    return panel_frames

# -------------------------------------------
# 10) 主流程（批量）
# -------------------------------------------
def process_patient_extra(mat_file_path, base_output_dir,
                          fullcorr_max_voxels=200_000,
                          min_brain_ratio=0.05,
                          slice_stride=3,
                          axes=('axial','coronal','sagittal'),
                          tiles_per_row=27,
                          max_voxels_curtain=50_000,
                          curtain_sort='mprage',
                          fading_K=20,
                          fading_fps=20):
    """
    输出：
      A) taskA_full_corr_351x351.png + .npz
      B) visualizations_panel/<axis>/full351_mosaic/<axis>_full351_sliceXXX.png
      C) visualizations_panel/curtain/taskC_curtain_modalities.png (+ .labels.txt)
      D) visualizations_panel/<axis>/fading_vsMPRAGE_all/<axis>_fading_all_vsMPRAGE_<abbr><slice>.gif
    """
    patient_name = mat_file_path.stem
    out_dir = Path(base_output_dir) / patient_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载
    d = load_minimal_3d_data(mat_file_path)
    data_4d = d['data']
    mask3d = d['region_mask'].astype(bool)

    # ---- A: 351×351 相关矩阵
    try:
        R = compute_full_corr_roi(data_4d, mask3d, max_voxels=fullcorr_max_voxels, random_seed=42)
        save_corr_heatmap(R, out_png=out_dir / 'taskA_full_corr_351x351.png',
                          title=f'ROI Pearson Corr 351×351 - {patient_name}')
        np.savez(out_dir / 'taskA_full_corr_351x351.npz', R=R)
    except Exception as e:
        print(f"[WARN] Full correlation failed: {e}")

    # ---- B: 一图351子图（按阈值后起步 + 每隔3层），注记=模态名
    viz_root = out_dir / "visualizations_panel"
    axis_map = {'sagittal':0, 'coronal':1, 'axial':2}
    axis_abbr = {'axial':'ax', 'coronal':'co', 'sagittal':'sa'}

    for axis_name in axes:
        (viz_root / axis_name / "full351_mosaic").mkdir(parents=True, exist_ok=True)
        (viz_root / axis_name / "fading_vsMPRAGE_all").mkdir(parents=True, exist_ok=True)
        ax = axis_map.get(axis_name, 2)
        n_slices = data_4d.shape[ax]
        start_idx = first_valid_slice(mask3d, ax, min_ratio=min_brain_ratio)
        if start_idx is None:
            print(f"[WARN] {axis_name}: 未找到满足脑占比阈值({min_brain_ratio})的切片，跳过。")
            continue

        for k in range(start_idx, n_slices, slice_stride):
            mask2d = get_slice_2d(mask3d, ax, k)
            if mask2d.mean() < min_brain_ratio:
                continue

            # (B) 351 面板（PNG）
            png_path = viz_root / axis_name / "full351_mosaic" / f"{axis_name}_full351_slice{k:03d}.png"
            try:
                save_full351_mosaic_for_slice(
                    data_4d, mask3d, axis=ax, slice_idx=k,
                    tiles_per_row=tiles_per_row, pad_px=2, annotation_h=22,
                    out_png=png_path
                )
                print(f"   ✓ [B] {axis_name} slice={k} 351面板已保存")
            except Exception as e:
                print(f"[WARN] 351面板失败 @{axis_name} slice {k}: {e}")

            # (D) 351维度对比（其他350 vs MPRAGE）GIF
            gif_path = viz_root / axis_name / "fading_vsMPRAGE_all" / f"{axis_name}_fading_all_vsMPRAGE_{axis_abbr[axis_name]}{k:03d}_K{fading_K}.gif"
            try:
                save_fading_panel_all_vs_mprage_for_slice(
                    data_4d, mask3d, axis=ax, slice_idx=k,
                    tiles_per_row=tiles_per_row, pad_px=2, annotation_h=22,
                    K=fading_K, fps=fading_fps, out_gif=gif_path
                )
                print(f"   ✓ [D] {axis_name} slice={k} 350×fading(GIF) 已保存")
            except Exception as e:
                print(f"[WARN] Fading GIF 失败 @{axis_name} slice {k}: {e}")

    # ---- C: Curtain（单幅）
    (viz_root / "curtain").mkdir(parents=True, exist_ok=True)
    try:
        save_modalities_curtain(
            data_4d, mask3d,
            out_png=viz_root / "curtain" / "taskC_curtain_modalities.png",
            max_voxels=max_voxels_curtain,
            sort_by=curtain_sort,
            dpi=300
        )
        print(f"   ✓ [C] Curtain 已保存")
    except Exception as e:
        print(f"[WARN] Curtain 生成失败: {e}")

    # ---- 元信息
    meta = {
        'subject': patient_name,
        'time': datetime.now().isoformat(),
        'ref_modality_idx': int(REFERENCE_MODALITY),
        'slice_stride': int(slice_stride),
        'min_brain_ratio': float(min_brain_ratio),
        'axes': list(axes),
        'tiles_per_row': int(tiles_per_row),
        'curtain_max_voxels': int(max_voxels_curtain),
        'curtain_sort': curtain_sort,
        'fading_K': int(fading_K),
        'fading_fps': int(fading_fps),
        'spacing_mm': SPACING,
        'notes': FALLBACKS
    }
    with open(out_dir / 'task_extra_v1.3.2_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Extra tasks v1.3.2 完成: {patient_name}")
    return out_dir

# -------------------------------------------
# 11) 批处理入口
# -------------------------------------------
def main():
    print("======= QC Extra v1.3.2 =======")
    print(f"DATA_DIR        : {DATA_DIR}")
    print(f"BASE_OUTPUT_DIR : {BASE_OUTPUT_DIR}")
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(list(Path(DATA_DIR).glob('*.mat')))
    if not mat_files:
        print(f"[WARN] 未在 {DATA_DIR} 找到 .mat 文件")
        return

    for f in mat_files:
        try:
            process_patient_extra(
                f, BASE_OUTPUT_DIR,
                fullcorr_max_voxels=200_000,
                min_brain_ratio=0.05,
                slice_stride=3,
                axes=('axial','coronal','sagittal'),
                tiles_per_row=27,            # 351/27≈13行；尽量保持宽幅高清
                max_voxels_curtain=50_000,
                curtain_sort='mprage',       # 可设 None
                fading_K=20,                 # 透明度切换帧数（与原风格一致）
                fading_fps=20
            )
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

if __name__ == "__main__":
    main()
