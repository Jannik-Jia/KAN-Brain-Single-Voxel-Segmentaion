# AttnResMLP 于 Alex7T-102labels

## 1. 在论文中的角色
- 面向 Alex 超多模态 7T 数据集（German et al. 2021）的体素级分类，作为带注意力的残差 MLP 基线。
- 在保持单体素输入的前提下，测试 transformer 式升级（Stage1 基础残差，Stage2 加自注意力，Stage3 加 pre-LN + FFN）相对简单 FC/KAN 方案的效果。
- 聚焦 102 个硬标签，带类别失衡缓解，评估 train/val/test 与 merged 集。

## 2. 代码文件与入口
- ：端到端 notebook，定义数据采样器、dataloader、可选 PCA、模型变体（Stage1–3）、训练循环、评估与可视化。
- 关键组件：、 、 、 、模型类 、 、 、 、 ，训练/评估辅助（ 、 ，绘图）。
- 输出保存在 notebook 相对路径下的 。

## 3. 数据集与标签
- 数据集：German et al. 2021 的 Alex 超多模态 7T；体素特征位于 。
- 输入维度：默认 341 通道（diffusion/QTI、CEST offsets、Amide/Amine/NOE/MT 等），可选 PCA 到 （默认 0 = 不做 PCA）。
- 标签：102 个组织/区域类别（id 1–102）+ 背景 0；背景重映射为 ，在 loss 中忽略；类别映射存到 。
- 划分：由  重构，大约 60/20/20 train/val/test；额外的“merged”集合载入所有标签做 sanity check。
- 采样：可选平衡（, ）并用简单复制增强，最多 ；也可使用全样本（）。

## 4. 预处理流程
- 可选在 train/test/val 拼接后做 PCA，再对每个主成分做 min-max 归一化；默认 （使用原始 341 维特征）。
- PCA 关闭时不做显式归一化；假设上游已做特征缩放。
- 标签处理：背景 -> ；类别权重由训练标签计数计算。
- 数据重构工具合并旧 train/val 并按 0.6/0.2/0.2 切分（上游一次性执行）。
- 无缺失值处理或模态特定缩放。

## 5. 模型结构
- Stage1 ：输入线性层 + GELU + BatchNorm -> 在  上堆叠 （默认 [256,256,256]）-> 线性分类头。
- Stage2：在每个残差块后加入多头 （num_heads=4, dropout=0.1），带 LayerNorm 与残差融合。
- Stage3 Transformer 风格：输入嵌入（Linear -> LayerNorm -> SiLU -> Dropout 重复），若干 pre-LN 多头注意力（num_heads=8, dropout=0.2）+ FFN（4x 扩张，SiLU）+ 投影残差，最终 LayerNorm + 分类器。
- 正则化：dropout 0.1/0.2，前期阶段含 BatchNorm；定义了 L1/entropy 正则标志（, , ），但未在损失中实际使用。

## 6. 训练配置
- 损失： 带逆频次类别权重， 背景忽略。
- 优化器：AdamW (, )；seed 666。
- 学习率调度：可选；默认  里程碑 [15,35,50,75]，gamma 0.6；也实现了 cosine 与 plateau 方案。
- Batch size 256；训练 100 轮；每 3 轮验证一次。
- 通过重采样/复制增强做数据平衡；无 mixup/cutmix；无显式梯度裁剪。
- 检查点每次验证保存到 ；配置保存到 。

## 7. 评估指标与输出
- 指标：accuracy、balanced accuracy、macro/weighted F1、Cohen κ；按类 precision/recall/F1；混淆矩阵（对数尺度热力图）。
- 可视化：类别分布柱状图、按类准确率柱状图、混淆矩阵、多数据集指标对比。
- 输出： 图（如 , , ），文本报告（, ），检查点 , , 。

## 8. 论文写作解读
- 为 Alex 数据集 102 类提供增强注意力和残差的强力 FC/MLP 基线。
- 展示 transformer 式组件如何影响体素级分类与类别失衡，为与 KAN/TabNet/线性基线的对比提供依据。
- 适合放在“基线与注意力增强 MLP”部分，或注意力深度的消融。

## 9. 限制与开放问题
- 精确指标数值未嵌入，需依赖已保存的报告。
- 数据路径硬编码到 ，且依赖预构建的 ，可移植性有限。
- 缺少校准指标（ECE/NLL）和不确定度估计；PCA 关闭时的归一化假设不明确。
- L1/entropy 正则标志未用；Stage1/2/3 对比未在 notebook 中给出。
