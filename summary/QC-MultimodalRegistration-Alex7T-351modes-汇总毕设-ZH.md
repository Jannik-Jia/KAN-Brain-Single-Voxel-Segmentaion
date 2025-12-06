# QC-MultimodalRegistration 于 Alex7T-351modes

## 1. 在论文中的角色
在训练前评估多模态配准/对齐的质量控制套件，确保 351 通道输入空间一致、无明显伪影。

## 2. 代码文件与入口
- `dataset_create/multimodal_mri_qc_analysis/multimodal_mri_qc_analysis.ipynb`：相似性分析主 notebook（任务 1）。
- `dataset_create/multimodal_mri_qc_analysis/multimodal_qc_tasks_extra.py`、`multimodal_mri_qc_analysis1.py`：任务 2–5 的脚本（边缘/ROI/QC 评分）。
- `dataset_create/multimodal_mri_qc_analysis/MULTIMODAL_QC_ANALYSIS_README.md`、`QUICKSTART_QC_ANALYSIS.md`：流程、阈值、数据需求。
- `dataset_create/multimodal_mri_qc_analysis/CHANGELOG_QC_v1.1.0.md`、`CHANGELOG_QC_v1.2.0.md`：算法与性能更新。

## 3. 数据集与标签
- 数据集：Alex 7T 的 3D “minimal” 体积（384×336×256×351），MPRAGE 间距约 0.65 mm；ROI 掩膜/标签与训练数据共用。
- 标签：使用 FreeSurfer 派生区域做 ROI 检查；此处不生成训练标签。

## 4. 预处理流程
- 读取 351 通道，按模态族分组，使用 0.65 mm 的间距敏感指标，可选 CEST slab 定位。
- 支持各向异性 PSF 考虑，并用掩膜限定脑实质体素。

## 5. 模型结构
- 不适用（纯分析/QC）。

## 6. 训练配置
- Notebook/脚本的阈值参数：LNCC/NGF 下限、MIND-SSD、ASSD/HD95 上限（mm）、edge IoU、基于 MAD 的异常检测。
- 可选 ROI 列表与 UMAP/PCA 的降维设置。

## 7. 评估指标与输出
- 相似性矩阵：LNCC、NGF、MIND-SSD（LNCC/NGF 越高越好，MIND-SSD 越低越好），跨模态族。
- 边缘一致性：ASSD、HD95（越低越好），edge IoU（越高越好），并用自适应 Canny 阈值。
- ROI 信号一致性热图；PCA/UMAP 散点；QC PASS/WARN/FAIL 评分与 CSV/JSON 摘要。
- 输出：PNG 图（相似性矩阵、边缘指标、ROI 热图、PCA/UMAP）、`modality_qc.csv`、`qc_analysis_report.json`，位于 `qc_analysis_results/`。

## 8. 论文写作解读
为多模态对齐质量提供客观证据，训练前识别问题通道。支撑数据整理与配准可靠性的章节，尤其在论证剔除质量差病例时。

## 9. 限制与开放问题
- 需要正确的数据路径与针对数据集的参数调优；对 351 通道全部评估时耗时较长。
- QC 阈值具启发性；PASS/WARN/FAIL 的判定可能需要人工核查。
