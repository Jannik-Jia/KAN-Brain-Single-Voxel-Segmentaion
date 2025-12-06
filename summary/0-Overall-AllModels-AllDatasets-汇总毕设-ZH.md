# 总体概要

该代码库覆盖了针对 Alex 超多模态 7T 数据集的体素级分类全流程：包含硬标签的 MLP 基线、软标签/校准实验、空间补丁 CNN（轻量版与 ResNet）、概率图后处理，以及多模态 QC/下采样工具。软标签实验强调校准与部分体积效应，空间模型验证邻域上下文的收益，QC 工具记录训练前的配准质量。

| ModelShort | DatasetShort | 毕设中的作用 | 关键脚本 | 关键指标 | 输出文件 |
| --- | --- | --- | --- | --- | --- |
| FC4x4096 | Alex7T-102labels | 硬标签基线与概率图生成器 | `training/B0_1D_training/train_1d_with_3d_dataset.py`; `erosion/train_38fold.py` | Macro-F1，整体准确率 | `.pth`，`predictions_3d` `.mat`，history JSON/ONNX |
| FC4x4096 | Alex7T-softlabels | 在下采样数据上的软标签与校准实验 | `training/downsampling/train_runner.py`; `dataset_create/downsampling/mri_downsampling_pipeline.py` | NLL、soft-ECE、Brier、AURC、soft Dice | `best.pth`、metrics JSON、可靠性/风险-覆盖率曲线、3D NPZ 预测 |
| ConvPatch2D | Alex7T-102labels | 轻量级空间基线（3×3/7×7 补丁） | `training/3D CNN/train_baseline_3x3_7x7.py` | Macro-F1 | `best_model_patch*.pth`，history JSON |
| ResNet50Patch | Alex7T-102labels | 使用失衡感知损失的深度空间模型 | `training/3D CNN/ResNet/scripts/train_mri_resnet.py` | Macro-F1，按类别 P/R/F1 | `best_model.pth`，`training_results.json`，`training_history.png` |
| ProbSmoothing | Alex7T-102labels | FC 概率图的后处理 | `training/B1_Probability_map_post_processing/smooth_postprocess_eval.py` | Accuracy、macro-F1、AUPRC、κ | 平滑后的 HDF5 预测、CSV 指标、图表 |
| QC-MultimodalRegistration | Alex7T-351modes | 多模态输入的配准/QC 分析 | `dataset_create/multimodal_mri_qc_analysis/multimodal_mri_qc_analysis.ipynb` | LNCC、NGF、MIND-SSD、ASSD/HD95、边缘 IoU | PNG 图、`modality_qc.csv`、`qc_analysis_report.json` |

论文撰写建议：将 FC4x4096（硬标签）作为主要基线；FC4x4096（软标签）用于软标签/校准分析；ConvPatch2D 与 ResNet50Patch 用于空间上下文消融；ProbSmoothing 用于后处理讨论；QC-MultimodalRegistration 用于数据质量/附录说明。注意：`HSIConvKAN-main` 含 KAN 参考实现，但未接入 MRI 数据流水线（用途未知）。
