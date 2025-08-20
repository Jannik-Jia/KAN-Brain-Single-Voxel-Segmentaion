# KAN-Brain MRI多模态脑区域分割数据集与3D体积数据转换系统

## 项目概述

本项目提供了一套完整的MRI多模态脑区域分割数据处理系统，包含38个被试的多模态MRI数据，用于深度学习驱动的脑区域分割任务。系统专门用于将FreeSurfer处理后的1D格式数据转换为3D体积数据格式，用于深度学习模型训练。每个被试的数据已经过FreeSurfer分割、配准和特征提取处理，包含高维多模态特征和精确的解剖标签。系统经过严格验证，确保数据转换的100%准确性。

## 核心功能

### 主要特性
- ✅ **100%验证准确性**: 所有数据转换都经过往返验证，确保无损转换
- 🚀 **批量处理能力**: 支持处理38个被试的大规模数据集
- 🛡️ **完整错误处理**: 包含断点续传、内存监控、优雅退出等企业级特性
- 📊 **详细日志记录**: 提供DEBUG级别的完整操作日志和进度追踪
- 🎯 **内存优化**: 使用直接布尔索引避免不必要的内存复制
- 📈 **性能监控**: 实时内存使用监控和性能统计

## 数据集详细说明

### 基本信息
- **被试数量**: 38个
- **文件格式**: MATLAB .mat文件 (HDF5格式)
- **平均文件大小**: 3.2GB（原始），转换后约200-250MB（压缩）
- **3D数据尺寸**: 384 × 336 × 256 体素
- **总体素数**: 33,030,144个
- **有效脑组织体素**: 约150万-240万个（因被试而异）
- **特征维度**: 351维多模态MRI特征
- **目标类别**: 102个脑区域

### **重要：数据存储格式说明**
**HDF5文件中数据使用MATLAB的Fortran顺序（列优先）存储，Python加载时需要转置**
- **原始存储**：MATLAB格式，使用Fortran顺序
- **Python处理**：需要转置以适配行优先习惯
- **正确加载方式**：多维数组必须转置（`.T`）

### **验证结论**
**经过严格验证确认的数据顺序规律：**
- **数据一致性**：所有1D数据（multidim_data, region_seg, seg_one_hot）使用相同的体素顺序

## 文件结构与数据变量

### 输入格式（1D MAT文件）

每个MAT文件包含5个关键数据变量，详细描述如下：

#### 1. `big_seg` - FreeSurfer完整分割结果

```
维度: (384, 336, 256)
数据类型: float64
数值范围: 0 - 5002
唯一值数量: ~194个
非零体素: ~3,800,000个
```

**描述**: 
FreeSurfer软件对整个脑体积进行解剖分割的完整结果，包含5002个不同的解剖标签，涵盖皮层、皮下结构、脑干、小脑等所有脑区域。

- `0`: 背景（非脑组织）
- `1-5002`: 不同的解剖结构（海马、杏仁核、各皮层区域等）

**用途**: 提供完整的解剖参考，用于理解脑结构的空间分布

#### 2. `region` - 脑组织二值掩膜

```
维度: (384, 336, 256)
数据类型: uint8
数值范围: 0 - 1
唯一值数量: 2个
非零体素: ~2,000,000个
```

**描述**: 
定义有效脑组织区域的二值掩膜，决定哪些体素参与后续分析。

- `0`: 背景区域（颅骨外、脑脊液、非脑组织等）
- `1`: 有效脑组织区域（参与分析的体素）

**关键作用**: 建立3D坐标与1D数据间的映射关系，不同被试的有效体素数量反映个体脑体积差异

#### 3. `multidim_data` - 多模态特征数据

```
维度: (351, n_voxels)
数据类型: float32
特征维度: 351维
体素数量: 因被试而异（~1,200,000 - 2,400,000）
```

**描述**: 
每个有效体素的351维多模态MRI特征向量，包含多种定量MRI参数。

**特征组成**:
- T1加权成像数据
- CEST (Chemical Exchange Saturation Transfer) M0参数
- QSM (Quantitative Susceptibility Mapping) 数值
- 包括但不限于各种定量MRI指标

**数据特性**:
- 数据已标准化和质量控制
- 与region_seg和seg_one_hot的体素顺序严格对应（100%验证）

#### 4. `region_seg` - 体素级FreeSurfer标签

```
维度: (1, n_voxels)
数据类型: float64
数值范围: 4 - 5002
唯一值数量: ~186个
```

**描述**: 
记录每个有效体素在FreeSurfer原始分割中的标签值，作为标签映射的中间步骤。

- 包含约186个不同的FreeSurfer标签（从5002个原始标签中筛选）
- 与`multidim_data`的体素顺序完全对应
- 保留每个体素的详细解剖信息

**映射关系（100%验证）**: 
```
region中每个=1的体素 → 转置前按顺序提取对应的big_seg标签值 → 存储在region_seg中
```

#### 5. `seg_one_hot` - 102脑区域One-Hot编码

```
维度: (102, n_voxels)
数据类型: uint8
激活区域: ~100个（102个中的100个）
```

**描述**: 
将FreeSurfer原始标签重新映射并编码为102个预定义脑区域的one-hot表示。

- 每个体素用102维one-hot向量表示其所属区域
- 每个体素只属于一个区域（每列只有一个1）
- 102个区域是根据神经科学需求定义的感兴趣区域（ROI）
- 不是所有区域在每个被试中都存在

**用途**: 机器学习模型的目标标签，用于脑区域分类任务

**验证**: 与multidim_data使用完全相同的体素顺序

### 输出格式（3D MAT文件）

| 变量名 | 输出形状 | 描述 |
|--------|----------|------|
| `big_seg` | (384, 336, 256) | FreeSurfer完整分割（5002个标签，直接复制） |
| `region_mask` | (384, 336, 256) | 脑组织掩膜 |
| `data` | (384, 336, 256, 351) | 4D多模态特征体积 |
| `region_labels` | (384, 336, 256) | 102类别整数标签（0-101，从seg_one_hot转换） |
| `region_seg_3d` | (384, 336, 256) | FreeSurfer原始标签3D重构（~186个标签，值4-5002） |
| `prob_idx` | (384, 336, 256) | 被试ID体积（1-38） |

### 标签映射流程

```
FreeSurfer标签(4-5002) → 标签映射表 → 目标区域(0-101) → One-hot编码(102维)
```

### 数据完整性验证（100%确认）

#### 维度一致性检查
- `region`中非零体素数 = `multidim_data`的列数 = `region_seg`的列数 = `seg_one_hot`的列数
- 所有1D数据的体素顺序严格对应（已验证）

#### 标签一致性检查  
- **映射关系验证**: region_seg中的标签值100%来自big_seg中region=1位置
- **One-hot编码验证**: seg_one_hot与region_seg使用相同的体素顺序

## 技术实现

### 核心算法

#### 1. 正确的数据加载方法
```python
def load_mat_h5_correct(mat_path):
    """正确加载MATLAB HDF5格式的MAT文件"""
    data = {}
    with h5py.File(mat_path, "r") as f:
        for k in f.keys():
            if not k.startswith("#"):
                v = f[k][()]
                # 关键：只转置需要转置的数据
                if k == 'multidim_data' and v.shape[0] == 351:
                    v = v.T  # (351, n_voxels) → (n_voxels, 351)
                elif k == 'region_seg':
                    v = v.flatten()  # (1, n_voxels) → (n_voxels,)
                data[k] = v
    return data
```

#### 2. 核心转换原理
```python
# ✅ 正确方法：使用直接布尔索引
region = data['region'].astype(bool)
data_4d = np.zeros((*region.shape, n_features), dtype=features.dtype)
data_4d[region] = features  # 直接布尔索引赋值

# ❌ 错误方法：使用Fortran order flatten
# 这会导致数据顺序错乱，匹配率仅0.76%
```

### 关键技术突破

1. **内存布局问题解决**: 发现并修正了Fortran order flatten导致的数据顺序错乱问题
2. **转置策略优化**: 只对需要转置的数据进行转置，避免不必要的操作
3. **直接布尔索引**: 使用NumPy的直接布尔索引进行数据映射，确保100%准确性
4. **压缩策略**: 对不同数据使用分级压缩策略，平衡文件大小和处理速度

### MATLAB与Python的差异处理
- **存储顺序**: MATLAB使用Fortran顺序（列优先），Python使用C顺序（行优先）
- **解决方案**: 加载时选择性转置，保存时使用Fortran数组格式
- **验证方法**: 往返转换测试确保100%准确性

### 内存优化策略
- 使用直接布尔索引避免flatten操作
- 及时释放不需要的变量
- 定期垃圾回收（gc.collect()）
- 分级压缩策略减少I/O

### 错误处理机制
- **断点续传**: 自动保存处理进度
- **优雅退出**: 捕获Ctrl+C信号，保存状态后退出
- **紧急报告**: 异常退出时生成详细状态报告
- **重试机制**: 网络或I/O错误时自动重试

## 使用指南

### 环境要求
```bash
# Python 3.7+
pip install numpy h5py pandas tqdm psutil
```

### 快速开始

#### 1. 测试单个被试
```python
python batch_convert_validated_with_logging.py --test-only
```

#### 2. 批量处理所有被试
```python
python batch_convert_validated_with_logging.py \
    --data-dir /path/to/1D/data \
    --output-dir /path/to/3D/output \
    --start 1 --end 38
```

#### 3. 断点续传
```python
# 程序会自动保存检查点，中断后再次运行即可继续
python batch_convert_validated_with_logging.py --start 1 --end 38
```

### 高级功能

#### 日志级别控制
```python
# 可选: DEBUG, INFO, WARNING, ERROR
python batch_convert_validated_with_logging.py --log-level DEBUG
```

#### 跳过已存在文件
```python
python batch_convert_validated_with_logging.py --skip-existing
```

#### 处理特定范围
```python
# 只处理被试5到10
python batch_convert_validated_with_logging.py --start 5 --end 10
```

### 生物信号维度说明

- **总维度数**: 351维多模态特征
- **特征类型**: 包括T1加权、CEST参数、QSM数值等定量MRI指标

## 验证系统

### 自动验证
每次转换都会自动进行以下验证：
- ✅ 特征数据完整性验证
- ✅ 标签数据一致性验证
- ✅ One-Hot编码正确性验证
- ✅ 3D体数据保持验证
- ✅ 掩膜数据一致性验证

### 验证结果
所有验证通过时显示：
```
✅ 被试 1 处理成功 (7.5秒, 245.3MB)
```

部分验证失败时显示：
```
⚠️ 被试 2 处理成功但验证未完全通过
    - multidim_data: 验证失败
```

## 输出文件

### 生成的文件结构
```
output_dir/
├── logs/
│   └── conversion_YYYYMMDD_HHMMSS.log  # 详细日志
├── dataset_index_logged.json           # 数据集索引
├── conversion_checkpoint.pkl           # 断点续传检查点
├── batch_report_YYYYMMDD_HHMMSS.json  # 批处理报告
├── batch_summary_YYYYMMDD_HHMMSS.csv  # CSV摘要
├── subject1_3d_validated.mat          # 转换后的3D数据
├── subject2_3d_validated.mat
└── ...
```

### 报告内容
- **JSON报告**: 包含所有处理细节、错误信息、验证结果
- **CSV摘要**: 便于Excel查看的处理统计
- **日志文件**: DEBUG级别的完整操作记录

## 性能指标

### 实测性能
- **处理速度**: 单个被试7-12分钟
- **总处理时间**: 38个被试约5-8小时
- **内存峰值**: 16-20GB
- **CPU使用**: 单核心（可优化为多进程）
- **磁盘I/O**: 读取~3GB，写入~250MB每被试

### 优化建议
- 使用SSD硬盘可显著提升I/O性能
- 建议至少16GB内存，32GB更佳
- 可并行处理多个被试（需修改代码）

## 故障排除

### 常见问题

#### 1. 内存不足
**症状**: `MemoryError`或系统变慢
**解决**: 
- 增加系统内存
- 减少并发处理数
- 使用`--test-only`先测试单个被试

#### 2. 文件未找到
**症状**: `FileNotFoundError`
**解决**:
- 检查路径是否正确
- 确认文件扩展名为`.mat`
- 验证文件权限

#### 3. 验证失败
**症状**: 验证匹配率低于100%
**解决**:
- 检查原始数据完整性
- 确认使用正确的加载函数
- 查看日志文件定位问题

#### 4. 程序中断
**症状**: 意外退出
**解决**:
- 程序支持断点续传，直接重新运行即可
- 检查`conversion_checkpoint.pkl`文件
- 查看`emergency_report_*.json`了解中断时状态

## 项目结构

```
3D_dev/
├── batch_convert_validated_with_logging.py  # 主程序（带日志版）
├── batch_convert_validated.py              # 基础版批量转换
├── README.md                               # 项目技术文档
├── README_merged.md                        # 本合并文档
├── mri_dataset_readme.md                   # 数据集详细说明
├── analysis_new.md                         # 新分析文档
├── analysis_old.md                         # 旧分析文档
├── project_completion_summary.md           # 项目总结
└── logs/                                   # 日志目录（自动创建）
```

## 版本历史

### v2.0.0 (2025-08-20)
- ✨ 添加详细日志记录系统
- ✨ 实现断点续传功能
- ✨ 添加内存监控
- ✨ 优雅退出和信号处理
- ✨ 紧急报告生成
- 🐛 修复所有已知数据转换问题
- 📝 更新为准确的文档

### v1.0.0 (2025-08-17)
- 初始版本
- 基础1D到3D转换功能
- 简单验证系统

## 贡献者

- 数据处理算法设计与实现
- 验证系统开发
- 文档编写

## 许可证

本项目仅供研究使用。使用前请确保获得数据集的使用许可。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue到项目仓库
- 查看日志文件获取详细错误信息
- 参考故障排除章节

---

*最后更新: 2025-08-20*
*版本: 2.0.0*
*状态: 生产就绪*