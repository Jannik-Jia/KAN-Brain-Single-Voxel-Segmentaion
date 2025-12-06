# ResNet50Patch 于 Alex7T-102labels

## 1. 在论文中的角色
更深的空间模型（约 50M 参数），检验丰富的空间处理与失衡感知损失是否能在 7×7 补丁上超越轻量 ConvPatch2D 与 MLP 基线。

## 2. 代码文件与入口
- `training/3D CNN/ResNet/scripts/train_mri_resnet.py`：主训练器，含损失/增强选项、早停与日志。
- `training/3D CNN/ResNet/models/resnet.py`：MRI 优化的 ResNet-50 定义（expand-first stem，base_width=104）。
- `training/3D CNN/ResNet/models/dataset.py`：补丁加载器，支持平衡/增强；`models/losses.py`：focal/类平衡/Logit 调整损失与 mixup 工具。
- `training/3D CNN/ResNet/configs/default_config.json`：默认超参；`scripts/run_leave_one_out.sh` 做批量 CV；`scripts/analyze_resnet_results.py` 汇总结果。

## 3. 数据集与标签
- 数据集：同 Alex 7T MAT（384×336×256×351），默认 7×7 补丁。
- 输入：每补丁 351 通道；可选类别平衡采样或加权采样。
- 标签：102 个硬类别（在加载器中移到 0–101）；掩膜去除背景。
- 划分：按受试留一；可选每受试采样（默认 1 万）或文档描述的“内存友好”全数据模式。

## 4. 预处理流程
- 每受试按通道 z-score；可缓存内存。
- 可选增强：随机翻转/旋转、轻 Gaussian 噪声（`--augmentation` 时）。
- 提供加权采样与类别权重以缓解失衡。

## 5. 模型结构
- ResNet-50（3-4-6-3 bottleneck），expand-first 3×3 stem（351→512），无 max pooling，base_width=104（~50M 参数）。
- 空间流：7→4→2→1；通道流：512→256→512→1024→2048→102 分类器。
- 可选 EMA 跟踪、mixup、梯度裁剪。

## 6. 训练配置
- 损失选项：CE、加权 CE、focal、类平衡 focal（默认 gamma=1.5, beta=0.9999）、logit-adjusted CE (tau)、balanced softmax；可选 label smoothing。
- 优化器 AdamW（lr=1e-4, weight_decay=1e-4）；CosineAnnealingLR（T_max=100, eta_min=1e-6）；梯度裁剪 1.0。
- Batch 256；训练 100 轮；早停 patience=15；可选 mixup (alpha=0.2) 与 EMA (decay=0.999)。

## 7. 评估指标与输出
- 指标：训练/测试 macro-F1，按类 precision/recall/F1/support；学习率与过拟合诊断。
- 输出：`best_model.pth`，每 10 轮检查点，`training_results.json`，`training_history.png`，`training.log`，位于 `resnet_test_subject_*` 目录。

## 8. 论文写作解读
检验更深空间建模与失衡感知目标是否优于简单补丁 CNN 与 MLP。结果为空间上下文、类别失衡处理及高级优化技巧（mixup/EMA）的消融提供依据。

## 9. 限制与开放问题
- 文档描述的“内存友好”全数据模式默认加载器仍采样；71M 体素全量训练的影响未验证。
- 无显式校准指标；损失超参（gamma/beta/tau）需调节。
- 补丁大小默认 7，改动需重训；数据路径需手动配置。
