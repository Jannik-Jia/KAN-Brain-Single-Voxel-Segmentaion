# CLAUDE.md - KAN-Brain-Single-Voxel-Segmentation 3D开发文档

## 项目概述

这是一个MRI多模态脑区域分割数据处理库，专门用于处理38个被试的多模态MRI数据。主要功能是将FreeSurfer处理后的1D格式数据重映射为3D体积数据，用于深度学习模型训练。

## 项目结构

```
3D_dev/
├── mri_dataset_readme.md      # 数据集详细文档
├── 3d_data_create.ipynb       # 核心1D→3D转换脚本
├── 3D_validate.ipynb          # 数据验证工具
├── 3d_data_sqeuence_order_test.ipynb  # 数据顺序测试
├── 38_prob_mat_analysis.ipynb # 38个被试数据分析
└── CLAUDE.md                  # 本文档
```

## 数据集信息

### 基本参数
- **被试数量**: 38个
- **3D数据尺寸**: 384 × 336 × 256 体素
- **总体素数**: 33,030,144个
- **有效脑组织体素**: 约150万-240万个（因被试而异）
- **特征维度**: 351维多模态MRI特征
- **目标类别**: 102个脑区域
- **文件大小**: 单个文件约3.2GB

### 数据变量说明

#### 原始1D格式（输入）
1. **big_seg** (384, 336, 256): FreeSurfer完整分割结果，5002个解剖标签
2. **region** (384, 336, 256): 脑组织二值掩膜，定义有效体素
3. **multidim_data** (n_voxels, 351): 多模态MRI特征数据
4. **region_seg** (1, n_voxels): 每个体素的FreeSurfer标签
5. **seg_one_hot** (102, n_voxels): 102脑区域的one-hot编码

#### 3D格式（输出）
1. **data** (384, 336, 256, 351): 4D多模态特征体积
2. **region** (384, 336, 256): 3D整数标签（0-101）
3. **prob_idx** (384, 336, 256): 被试ID体积（1-38）
4. **big_seg** (384, 336, 256): 原始FreeSurfer分割（验证用）

## 核心功能模块

### 1. 数据加载模块 (load_mat_h5)

处理MATLAB与Python之间的数据格式差异：
- MATLAB使用Fortran顺序（列优先）
- Python使用C顺序（行优先）
- 自动转置多维数组以适配Python

```python
def load_mat_h5(path):
    """正确加载MATLAB HDF5格式的MAT文件"""
    data = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if not k.startswith("#"):
                v = f[k][()]
                if v.ndim > 1:
                    v = v.T  # 关键：转置多维数组
                data[k] = v
    return data
```

### 2. 1D到3D重映射模块

#### 基础版本 (revert_reshape_f_order)
- 使用Fortran顺序将1D数组重构回3D体积
- 保持与原始MATLAB逻辑一致
- 支持单特征或多特征处理

#### 批量优化版本 (revert_reshape_f_order_batch)
- 使用向量化操作处理所有特征
- 比逐个特征处理快100倍以上
- 用于处理351维特征数据

### 3. 数据处理流程 (process_single_subject_to_3d)

处理步骤：
1. 加载原始MAT文件（自动处理转置）
2. 将multidim_data重映射为4D体积
3. 将seg_one_hot转换为3D整数标签
4. 创建prob_idx的3D体积
5. 保存为HDF5格式的MAT文件
6. 验证数据完整性

### 4. 数据验证系统 (DatasetValidator)

验证项目：
- **维度验证**: 确保3D数据形状正确
- **big_seg复制**: 验证原始分割数据完整性
- **prob_idx验证**: 确保被试ID正确
- **multidim_data重映射**: 验证特征数据映射正确性
- **标签转换**: 验证one-hot到整数标签的转换
- **数据完整性**: 通过逆向还原验证
- **统计特性**: 验证数值范围和分布

### 5. 可视化工具

提供多种可视化功能：
- 切片级别的特征可视化
- 多个生物信号对比
- 验证结果可视化
- 生成验证报告

## 关键技术点

### 1. 数据存储格式

**问题**: MATLAB与Python的多维数组存储顺序不同
- MATLAB: Fortran顺序（列优先）
- Python: C顺序（行优先）

**解决方案**: 
- 加载时转置多维数组
- 使用Fortran顺序进行重构操作
- 保存时转置回MATLAB格式

### 2. 内存优化

**挑战**: 单个4D数组约46GB（384×336×256×351×4字节）

**优化策略**:
- 使用向量化操作减少内存复制
- HDF5压缩存储（gzip）
- 分级压缩策略（data用低压缩率）

### 3. 数据一致性保证

**验证方法**:
- 重构-还原循环测试
- 统计特性比较
- 可视化抽查
- 100%匹配率要求

## 使用指南

### 基本使用流程

```python
# 1. 创建数据集索引
index_mapping = create_index_json(data_dir)

# 2. 批量处理所有被试
successful, failed = process_all_subjects(data_dir, output_dir)

# 3. 验证处理结果
validator = DatasetValidator(json_path, original_dir, remapped_dir)
results = validator.validate_all()

# 4. 可视化检查
validator.spot_check_visualization(prob_idx=1, slice_idx=210)
```

### 文件路径配置

```python
# 标准路径配置
data_directory = "/home/jannik/Documents/mri_mat_onehot/1D"  # 原始1D数据
output_3d_dir = "/home/jannik/Documents/mri_mat_onehot/3D"   # 3D输出目录
json_path = "/home/jannik/Documents/mri_mat_onehot/1D/dataset_index.json"
```

## 性能指标

- **处理速度**: 单个被试约12-15分钟
- **总处理时间**: 38个被试约8-10小时
- **内存需求**: 建议至少16GB RAM
- **存储需求**: 输出文件每个约200-250MB（压缩后）
- **验证速度**: 全部验证约30分钟

## 常见问题

### Q1: 数据维度不匹配
**原因**: MATLAB格式未正确转置
**解决**: 确保使用load_mat_h5函数加载数据

### Q2: 内存溢出
**原因**: 同时处理多个大文件
**解决**: 逐个处理被试，使用数据生成器

### Q3: 验证失败
**原因**: 重映射逻辑错误或数据损坏
**解决**: 使用验证工具定位问题，检查特定被试

## 更新日志

### 2025-01-17
- 初始版本创建
- 实现1D到3D重映射功能
- 添加完整验证系统
- 优化批量处理性能

## 待办事项

- [ ] 添加多进程并行处理
- [ ] 支持增量处理（断点续传）
- [ ] 添加更多可视化选项
- [ ] 优化内存使用
- [ ] 添加命令行接口

## 注意事项

1. **数据完整性**: 处理前务必备份原始数据
2. **存储空间**: 确保有足够的磁盘空间（至少20GB）
3. **Python环境**: 需要numpy, h5py, matplotlib, tqdm等库
4. **验证步骤**: 处理后必须运行验证确保数据正确

## 联系信息

如有问题或需要支持，请：
1. 查看mri_dataset_readme.md获取更多数据集细节
2. 运行验证工具诊断问题
3. 检查生成的validation_report.json

---
*文档最后更新: 2025-01-17*