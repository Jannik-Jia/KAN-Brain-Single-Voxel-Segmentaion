"""
配置参数模块
"""
import os

# 基本配置
RANDOM_SEED = 666
LABEL_ID = 1  # 可以根据需要修改

# 数据预处理参数
APPLY_PCA = True   # 是否应用PCA降维
N_PCA = 24          # 设为0表示自动选择主成分数量，大于0表示使用指定数量
NORM = True        # 是否对数据进行标准化/归一化处理

# 采样策略参数
SAMPLING_STRATEGY = 'modified_stratified'  # 采样策略：'balanced', 'stratified', 'modified_stratified', 'hard_negative'
NEG_POS_RATIO = 5.0  # 负样本与正样本的比例
NEG_LABEL_COUNT = None  # 使用的负类标签数量，None表示使用所有

# 定义模型名称，用于结果保存和模型标识
MODEL_NAME = 'BrainVoxel_1DKAN'

# 指定数据集名称
DATASET = 'BrainVoxel'

# 训练参数
EPOCH = 100        # 总训练轮数
VAL_EPOCH = 1      # 每隔多少轮进行一次验证
LR = 0.001         # 学习率
WEIGHT_DECAY = 1e-6  # 权重衰减系数，用于L2正则化
BATCH_SIZE = 640    # 批处理大小，固定不变

# 计算设备选择
DEVICE = 0         # -1表示使用CPU，0表示使用第一块GPU(cuda:0)

# 数据参数
FEATURE_DIM = 341  # 输入特征维度
NUM_CLASS = 2      # 二分类问题：正类和负类
FIXED_GRID = 10     # 固定网格大小，不进行网格扩展

# 数据目录路径
DATA_DIRS = {
    'train_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/train",
    'test_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/test",
    'val_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/val",
    'merged_dir': "/home/jovyan/gpu_space/workspace_jiayi/KAN training/brain_voxel_data/restructured/merged"
}

# 模型检查点路径
CHECK_POINT = None  # 加载预训练模型的路径，None表示从头开始训练

# 结果保存路径
SAVE_PATH = f"./Results/{MODEL_NAME}/{DATASET}"
if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# 评估特定epoch
START_EPOCH = 1
END_EPOCH = 100
EVALUATION_METRICS = ['accuracy', 'f1', 'auc_pr', 'recall']
SAVE_RESULTS = True
PLOT_RESULTS = True

# 阈值设置
THRESHOLD_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
