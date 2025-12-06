# FC4x4096 于 Alex7T-softlabels

## 1. 在论文中的角色
在下采样到 CEST 分辨率的数据上，用概率标签训练 4×4096 网络，利用部分体积信息并评估校准/不确定性处理。

## 2. 代码文件与入口
- `training/downsampling/train_runner.py`：单划分训练器（36 训 / 1 验 / 1 测），带软标签指标与 3D 重建。
- `training/downsampling/notebook_quickstart.ipynb`、`training/downsampling/notebook_loso_37fold.ipynb`、`training/downsampling/NOTEBOOK_VERIFICATION.md`：基于 notebook 的运行与验证步骤。
- `dataset_create/downsampling/mri_downsampling_pipeline.py`、`dataset_create/downsampling/DOWNSAMPLED_DATA_FORMAT.md`：使用模态特定 PSF/间距生成带软标签的 CEST 分辨率 NPZ。
- `dataset_create/1d-3d-convert/data_3d_1d_mapper.py`：通用 3D↔1D 转换器，保持 C 序体素对齐。

## 3. 数据集与标签
- 数据集：Alex 超多模态 7T 数据下采样至 CEST 分辨率（≈1.8×1.8×3.0 mm³），存为成对 1D/3D NPZ。
- 输入：每体素 351 通道特征（`multidim_data`，形状 (n_vox, 351)）。
- 标签：102 类概率标签（`seg_one_hot` / `proba_labels`），保留部分体积信息；`region` 掩膜背景。
- 划分：固定 36/1/1 受试划分（train/val/test）；CLI 可选覆盖。

## 4. 预处理流程
- 每受试在 351 通道上 z-score；可选限制每受试体素数以控内存。
- 下采样流水线应用模态族特定 PSF/重采样并重计算参数图；掩膜保持 ROI 对齐。
- Data3D1DMapper 在保持体素顺序的前提下恢复 3D softmax 体积；记录各受试的归一化统计。

## 5. 模型结构
- 稠密 4×4096 MLP，Dropout(0.5) + ReLU；输出 102 维 logits。
- 对权重施加 1e-5 手动 L2；无 batchnorm 或注意力。

## 6. 训练配置
- 损失：针对概率标签的软交叉熵，可选类别权重（alpha=0.5），含 L2 正则与梯度裁剪（默认 1.0）。
- 优化器 Adam（lr=1e-5，weight_decay 由手动 L2 提供）；默认 batch 256；默认训练 3 轮（快速跑），以验证 NLL 选最佳检查点。
- 训练后做温度缩放以校准；默认随机种子 42。

## 7. 评估指标与输出
- 指标：NLL、整体准确率、macro/micro/weighted F1、top-k 准确率、Brier 分数及 Murphy 分解、class-mass error、soft-ECE（按类与均值）、AURC（风险-覆盖）、熵直方图。
- 校准：可靠性图、温度缩放摘要、软混淆矩阵。
- 3D 指标：soft Dice（macro）、ROI 内 3D NLL/Brier、恢复后的切片图。
- 输出：最佳模型 `best.pth`，`metrics_{val,test}.json`，`temperature_scaling.json`，`run_summary.json`，混淆 CSV，可靠性/风险-覆盖/熵图，3D 预测（`val_*/test_*_pred_softmax_3d.npz`）。

## 8. 论文写作解读
检验概率标签与校准是否能相对硬标签基线提升体素预测与不确定性估计。流水线突出校准指标（soft-ECE、AURC、Brier），并能报告 3D 概率质量（soft Dice），为“软标签与校准”章节提供依据。

## 9. 限制与开放问题
- 默认训练很短（epochs=3）；可能需要更长训练以达峰值表现。
- 假设下采样 NPZ 与一致的 CEST 分辨率掩膜已存在。
- 未做完整留一；结果仅代表一次 36/1/1 划分，除非手动扩展。
