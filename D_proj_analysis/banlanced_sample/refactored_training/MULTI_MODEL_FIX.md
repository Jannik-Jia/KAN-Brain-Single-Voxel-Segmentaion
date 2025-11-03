# 多模型训练脚本修复总结

## 问题描述

KAN模型训练成功完成，但在整理输出文件时出现问题：

1. **符号链接删除失败**: `rm: cannot remove './results': Is a directory`
2. **文件统计为0**: 所有文件计数显示为0（模型文件、3D Softmax、训练曲线等）

```
✓ 模型文件: 0
✓ 3D Softmax: 0
✓ 训练曲线: 0
✓ Per-class分析: 0
```

但从训练日志可以看到文件确实被保存了：
```
💾 3D softmax已保存: results/test_softmax_3d_FOR_016_...nii.gz (168.87 MB)
💾 训练图表已保存: results/training_history_kan_bg_excl_20251102_163819.png
💾 模型已保存: results/kan_bg_excl_20251102_163819.pth
```

---

## 问题分析

### 问题1: 符号链接删除失败

**原因**: 脚本使用 `rm "$old_results_dir"` 来删除符号链接，但如果 `./results` 由于某些原因变成了真实目录，`rm` 命令会失败并报错。

**影响**: 阻止了后续的清理和文件整理操作。

### 问题2: 文件统计为0

可能的原因：
1. **符号链接路径问题**: 使用相对路径创建符号链接可能导致路径解析错误
2. **wc -l 格式问题**: `wc -l` 输出包含前导空格
3. **时序问题**: 删除符号链接后，文件可能不在预期位置

---

## 修复方案

### 修复1: 改进符号链接删除逻辑 ✅

**修改位置**: `run_multiple_models.sh` 第316-320行和第333-337行

**修改前**:
```bash
rm "$old_results_dir"
```

**修改后**:
```bash
if [[ -L "$old_results_dir" ]]; then
    rm "$old_results_dir"
elif [[ -d "$old_results_dir" ]]; then
    rm -rf "$old_results_dir"
fi
```

**说明**: 先检查是符号链接还是目录，然后使用相应的删除命令。

### 修复2: 使用绝对路径创建符号链接 ✅

**修改位置**: `run_multiple_models.sh` 第305-313行

**修改前**:
```bash
ln -sf "$temp_results_dir" "$old_results_dir"
```

**修改后**:
```bash
# 创建目标results目录
mkdir -p "$temp_results_dir"

# 创建符号链接（使用绝对路径）
local abs_temp_results_dir=$(cd "$(dirname "$temp_results_dir")" && pwd)/$(basename "$temp_results_dir")
ln -sf "$abs_temp_results_dir" "$old_results_dir"

# 验证符号链接
if [[ ! -L "$old_results_dir" ]]; then
    echo -e "${RED}❌ 创建符号链接失败${NC}"
    return 1
fi
```

**说明**: 使用绝对路径避免相对路径解析问题，并添加验证。

### 修复3: 改进文件统计和添加调试信息 ✅

**修改位置**: `run_multiple_models.sh` 第371-384行

**修改前**:
```bash
local pth_count=$(find "$results_dir" -name "*.pth" 2>/dev/null | wc -l)
```

**修改后**:
```bash
local pth_count=$(find "$results_dir" -name "*.pth" 2>/dev/null | wc -l | tr -d ' ')

# 调试信息
echo -e "${YELLOW}  调试: 检查目录 $results_dir${NC}"
echo -e "${YELLOW}  调试: 目录内容:${NC}"
ls -la "$results_dir" 2>/dev/null | head -10
```

**说明**:
- 使用 `tr -d ' '` 去除 `wc -l` 输出的空格
- 添加调试信息显示实际的目录内容

---

## 手动恢复KAN训练结果

如果KAN模型已经训练完成但文件没有正确整理，可以手动恢复：

### 步骤1: 查找训练结果

```bash
cd /Users/jannik/KAN-Brain-Single-Voxel-Segmentaion/D_proj_analysis/banlanced_sample/refactored_training

# 查找可能的results目录
find . -name "results" -type d

# 查找KAN的pth文件
find . -name "kan_bg_*.pth"

# 查找KAN的nii.gz文件
find . -name "*kan*.nii.gz"
```

### 步骤2: 识别正确的输出目录

从日志中我们知道：
- 模型保存时间戳: `20251102_163819`
- 训练运行目录: `./training_runs/kan_bg_excl_20251102_162156/`

```bash
# 检查这个目录
ls -la ./training_runs/kan_bg_excl_20251102_162156/results/
```

### 步骤3: 手动移动文件（如果需要）

如果文件在错误的位置，手动移动：

```bash
# 示例：如果文件在当前目录的results下
TARGET_DIR="./training_runs/kan_bg_excl_20251102_162156/results"

# 移动模型文件
mv results/kan_bg_excl_*.pth "$TARGET_DIR/" 2>/dev/null

# 移动3D softmax
mv results/test_softmax_3d_*_bg_excl_*.nii.gz "$TARGET_DIR/" 2>/dev/null
mv results/test_softmax_info_*.json "$TARGET_DIR/" 2>/dev/null

# 移动训练曲线
mv results/training_history_kan_*.png "$TARGET_DIR/" 2>/dev/null

# 移动per-class分析（如果有）
mv results/per_class_analysis_kan_* "$TARGET_DIR/" 2>/dev/null
```

### 步骤4: 验证文件

```bash
# 列出所有文件
ls -lh ./training_runs/kan_bg_excl_20251102_162156/results/

# 应该看到：
# - kan_bg_excl_*.pth (模型文件)
# - test_softmax_3d_*.nii.gz (3D softmax)
# - test_softmax_info_*.json (softmax信息)
# - training_history_kan_*.png (训练曲线)
# - per_class_analysis_kan_*/ (per-class分析目录，可选)
```

---

## 测试修复

### 重新运行训练（推荐）

使用修复后的脚本重新训练一个快速测试：

```bash
cd refactored_training

# 快速测试（5 epochs）
EPOCHS=5 ./run_multiple_models.sh
# 选择: 4 (kan)

# 检查输出
ls -la ./training_runs/kan_bg_excl_*/results/
```

**预期输出**:
```
调试: 检查目录 ./training_runs/kan_bg_excl_20251102_XXXXXX/results
调试: 目录内容:
total XXX
drwxr-xr-x  kan_bg_excl_*.pth
drwxr-xr-x  test_softmax_3d_*.nii.gz
drwxr-xr-x  training_history_*.png
...

✓ 模型文件: 1
✓ 3D Softmax: 1
✓ 训练曲线: 1
✓ Per-class分析: 0  # 如果没有visualization_toolkit
```

### 完整训练

确认修复后，运行完整训练：

```bash
# KAN完整训练（25 epochs）
EPOCHS=25 ./run_multiple_models.sh
# 选择: 4

# 或直接使用Python
python train.py --model kan --epochs 25
```

---

## 当前状态检查

检查之前的KAN训练是否有可用结果：

```bash
# 查看训练运行目录
ls -la ./training_runs/kan_bg_excl_*/

# 查看日志
cat ./training_runs/kan_bg_excl_*/logs/training_*.log | tail -50

# 查看汇总
cat ./training_runs/kan_bg_excl_*/SUMMARY.txt
```

如果文件都在，可以直接使用。如果文件不完整，建议重新训练。

---

## 相关文件

### 修改的文件
1. `run_multiple_models.sh` - 修复符号链接和文件统计问题

### 相关文档
2. `MULTI_MODEL_GUIDE.md` - 多模型训练指南
3. `MULTI_MODEL_SUMMARY.md` - 功能总结
4. `KAN_SETUP.md` - KAN模型设置
5. `DEEP_MLP_FIX.md` - DeepMLP修复文档

---

## 总结

### 已修复的问题
1. ✅ 符号链接删除失败
2. ✅ 符号链接路径问题
3. ✅ 文件统计空格问题
4. ✅ 添加调试信息

### 待验证
- [ ] 重新运行训练验证修复
- [ ] 检查文件正确性
- [ ] 确认所有输出都保存正确

### 建议
1. **短期**: 手动恢复现有的KAN训练结果（如果文件存在）
2. **中期**: 使用修复后的脚本重新运行快速测试
3. **长期**: 完整重新训练所有模型，确保一致性

---

**更新时间**: 2025-11-02
**状态**: ✅ 已修复
**测试**: 待验证
