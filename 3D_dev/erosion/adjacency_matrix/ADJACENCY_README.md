# 3D脑区邻接矩阵计算系统

用于计算3D脑区域间邻接关系的完整系统，支持与混淆矩阵进行Hadamard乘积分析，识别"高邻接×高混淆"的区域对。

## 🎯 系统功能

### 核心功能
- ✅ **标准化邻接矩阵**: 生成固定大小（默认102×102）的标准化邻接矩阵，确保与混淆矩阵维度对齐
- ✅ **3D邻接矩阵计算**: 基于空间接触关系计算邻接矩阵，缺失标签对应行/列为零
- ✅ **智能标签检测**: 自动识别0-101、1-102、0-102等不同标签格式
- ✅ **详细数据验证**: 完整的标签范围、缺失标签、数据质量检查
- ✅ **高效存储格式**: HDF5格式，包含完整验证信息，支持小块读取
- ✅ **详细报告生成**: 每个数据集生成JSON和Markdown说明文件
- ✅ **批量处理**: 支持38个被试的自动化批量计算
- ✅ **混淆矩阵分析**: Hadamard乘积和rank相关性分析
- ✅ **可视化工具**: 邻接关系可视化和统计分析

### 邻接定义
- **6连通**: 面相邻（默认，较保守）
- **18连通**: 面和边相邻
- **26连通**: 面、边和顶点相邻（较宽松）

### 标准化矩阵空间
为了确保邻接矩阵能够与混淆矩阵进行正确的数学运算（如Hadamard乘积），所有患者的邻接矩阵都被标准化到固定的维度：

- **标准矩阵大小**: 默认102×102（对应标签0-101）
- **缺失标签处理**: 如果某个患者缺少某些脑区标签，对应的行和列将全部为零
- **维度对齐**: 保证邻接矩阵与混淆矩阵的索引完全一致，便于后续分析
- **一致性保证**: 所有患者使用相同的"主标签集"，确保跨患者比较的有效性

## 📁 文件结构

```
3D_dev/erosion/adjacency_matrix/
├── compute_adjacency_matrices.py       # 主计算脚本（增强版）
├── adjacency_analysis_utils.py         # 分析工具类
├── run_adjacency_batch.sh             # 批量处理脚本
├── test_adjacency_computation.py      # 测试和验证脚本
├── quick_adjacency_test.py            # 快速系统验证
├── demo_enhanced_adjacency.py         # 增强功能演示
├── ADJACENCY_README.md                # 使用说明文档
└── results/                           # 输出目录（自动创建）
    ├── subject1_3d_validated_adjacency_conn6.h5     # HDF5邻接矩阵
    ├── subject1_3d_validated_adjacency_detailed_report.json  # 详细JSON报告
    ├── subject1_3d_validated_adjacency_report.md    # 易读Markdown报告
    ├── subject2_3d_validated_adjacency_conn6.h5
    ├── subject2_3d_validated_adjacency_detailed_report.json
    ├── subject2_3d_validated_adjacency_report.md
    ├── ...
    ├── batch_adjacency_report_*.json                 # 批处理总报告
    └── adjacency_computation_*.log                   # 详细日志
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install numpy h5py scipy pandas matplotlib seaborn tqdm

# 确保有权限访问3D数据
cd "3D_dev/erosion/adjacency_matrix"
```

### 2. 快速系统验证
```bash
# 快速验证系统功能（推荐先运行）
python quick_adjacency_test.py \
    --data_dir /path/to/3d_validated

# 查看增强功能演示
python demo_enhanced_adjacency.py
```

### 3. 测试单个被试
```bash
# 测试计算功能（包含详细验证）
python compute_adjacency_matrices.py \
    --data_dir /path/to/3d_validated \
    --output_dir ./results \
    --test_only \
    --verbose
```

### 4. 批量处理所有被试
```bash
# 使用批处理脚本（推荐）
bash run_adjacency_batch.sh

# 或手动运行（包含标准矩阵大小设置）
python compute_adjacency_matrices.py \
    --data_dir /path/to/3d_validated \
    --output_dir ./results \
    --connectivity 6 \
    --standard_matrix_size 102 \  # 设置标准矩阵大小（默认102）
    --start_subject 1 \
    --end_subject 38
```

### 4. 分析结果
```bash
# 测试和可视化
python test_adjacency_computation.py \
    --adjacency_dir ./results \
    --visualize
```

## ⚙️ 详细配置

### 计算参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--connectivity` | 6 | 连通性定义：6/18/26 |
| `--data_dir` | - | 3D数据目录（必须） |
| `--output_dir` | ./results | 输出目录 |
| `--start_subject` | 1 | 起始被试编号 |
| `--end_subject` | 38 | 结束被试编号 |

### 性能优化

**内存使用**:
- 单个被试: ~2-4GB内存峰值
- 计算时间: 每个被试约1-3分钟
- 存储空间: 每个文件约1-5MB

**连通性选择**:
```bash
# 保守（推荐用于分析）
--connectivity 6

# 标准
--connectivity 18

# 宽松
--connectivity 26
```

## 📊 输出格式

### 文件命名策略

**完全基于原始文件名，确保一一对应**：

| 原始数据文件 | 邻接矩阵文件 | JSON报告 | Markdown报告 |
|-------------|-------------|----------|--------------|
| `subject1_3d_validated.mat` | `subject1_3d_validated_adjacency_conn6.h5` | `subject1_3d_validated_adjacency_detailed_report.json` | `subject1_3d_validated_adjacency_report.md` |
| `subject2_3d_validated.mat` | `subject2_3d_validated_adjacency_conn6.h5` | `subject2_3d_validated_adjacency_detailed_report.json` | `subject2_3d_validated_adjacency_report.md` |
| `patient_A_validated.mat` | `patient_A_validated_adjacency_conn6.h5` | `patient_A_validated_adjacency_detailed_report.json` | `patient_A_validated_adjacency_report.md` |

**命名规则**：
- HDF5文件: `{原始文件名}_adjacency_conn{连通性}.h5`
- JSON报告: `{原始文件名}_adjacency_detailed_report.json`
- Markdown报告: `{原始文件名}_adjacency_report.md`

### HDF5文件结构
```
subject1_3d_validated_adjacency_conn6.h5
├── adjacency/
│   ├── matrix              # 完整邻接矩阵 (标准化大小，默认102×102)
│   └── upper_triangle      # 上三角形式
├── contact_counts/
│   ├── counts              # 接触体素计数 (102, 102)
│   └── upper_triangle      # 上三角形式
├── sparse/
│   ├── row_indices         # 稀疏格式行索引
│   ├── col_indices         # 稀疏格式列索引
│   ├── adjacency_values    # 邻接值
│   └── contact_values      # 接触计数
└── metadata/
    ├── total_regions       # 总区域数
    ├── total_adjacencies   # 总邻接对数
    ├── adjacency_density   # 邻接密度
    ├── avg_contact_voxels  # 平均接触体素数
    └── unique_labels       # 存在的区域标签
```

### 批处理报告
```json
{
  "total_files": 38,
  "successful": 38,
  "failed": 0,
  "avg_processing_time": 95.3,
  "total_processing_time": 3621.4
}
```

## 🔬 分析工具使用

### 基本读取
```python
from adjacency_analysis_utils import AdjacencyMatrixReader

# 加载邻接矩阵
reader = AdjacencyMatrixReader('subject1_3d_validated_adjacency_conn6.h5')

# 获取完整矩阵
adjacency_matrix = reader.load_full_adjacency_matrix()

# 高效子矩阵读取
submatrix = reader.load_adjacency_submatrix([0, 1, 2], [3, 4, 5])

# 获取邻接对
pairs = reader.get_adjacency_pairs()
```

### 混淆矩阵分析
```python
from adjacency_analysis_utils import ConfusionAdjacencyAnalyzer

# 创建分析器
analyzer = ConfusionAdjacencyAnalyzer()

# Hadamard乘积
hadamard_product = analyzer.compute_hadamard_product(
    confusion_matrix, adjacency_matrix
)

# 找到高混淆邻接对
high_pairs = analyzer.find_high_confusion_adjacency_pairs(
    confusion_matrix, adjacency_matrix, threshold=0.1
)

# Rank相关性
correlation, p_value = analyzer.compute_rank_correlation(
    confusion_matrix, adjacency_matrix
)

# 完整分析
results = analyzer.analyze_adjacency_confusion_relationship(
    confusion_matrix, adjacency_matrix
)
```

### 批量分析
```python
from adjacency_analysis_utils import load_multiple_adjacency_matrices

# 批量加载
readers = load_multiple_adjacency_matrices('./results/')

# 对每个被试分析
for subject_id, reader in readers.items():
    adjacency_matrix = reader.load_full_adjacency_matrix()
    # 与对应的混淆矩阵进行分析...
```

## 📈 实际应用示例

### 识别高邻接×高混淆区域对

```python
# 1. 加载训练结果的混淆矩阵
confusion_matrix = np.load('confusion_matrix_subject1.npy')

# 2. 加载对应的邻接矩阵
reader = AdjacencyMatrixReader('subject1_3d_validated_adjacency_conn6.h5')
adjacency_matrix = reader.load_full_adjacency_matrix()

# 3. 分析
analyzer = ConfusionAdjacencyAnalyzer()
analysis = analyzer.analyze_adjacency_confusion_relationship(
    confusion_matrix, adjacency_matrix
)

# 4. 查看结果
print("高混淆邻接对:")
for pair in analysis['high_confusion_adjacency_pairs'][:10]:
    print(f"  区域{pair['region1']} ↔ 区域{pair['region2']}: "
          f"混淆值={pair['confusion_value']:.3f}")

# 5. 相关性分析
corr_info = analysis['rank_correlation']
print(f"Spearman相关性: r={corr_info['spearman_correlation']:.3f}, "
      f"p={corr_info['p_value']:.3f}")
```

### 跨被试统计分析

```python
# 统计所有被试的邻接密度
densities = []
for subject_id, reader in readers.items():
    metadata = reader.metadata
    densities.append(metadata['adjacency_density'])

print(f"邻接密度统计: 平均={np.mean(densities):.3f}, "
      f"标准差={np.std(densities):.3f}")
```

## 🐛 故障排除

### 常见问题

#### 1. 内存不足
**症状**: `MemoryError`
**解决**:
```bash
# 使用更保守的连通性
--connectivity 6

# 检查系统内存
free -h
```

#### 2. 数据加载失败
**症状**: `FileNotFoundError` 或 `KeyError`
**解决**:
```bash
# 检查数据文件格式
python -c "
import h5py
with h5py.File('subject1_3d_validated.mat', 'r') as f:
    print(list(f.keys()))
"

# 确认路径正确
ls /path/to/3d_validated/*.mat
```

#### 3. 计算结果异常
**症状**: 邻接密度过高/过低
**解决**:
```bash
# 运行测试脚本
python test_adjacency_computation.py --adjacency_file result.h5 --visualize

# 检查连通性设置
# 6连通 < 18连通 < 26连通
```

#### 4. HDF5文件损坏
**症状**: 读取错误
**解决**:
```bash
# 重新计算该文件
python compute_adjacency_matrices.py \
    --data_dir /path/to/data \
    --start_subject 1 \
    --end_subject 1  # 只重新计算一个被试
```

## 📊 预期结果

### 典型邻接统计
- **总区域数**: ~100个（并非所有102个区域都存在）
- **邻接对数**: 300-800对（取决于连通性）
- **邻接密度**: 0.06-0.15（6连通到26连通）
- **平均接触体素**: 50-500个

### 与混淆矩阵的关系
- **正相关**: 邻接区域更容易被混淆
- **相关系数**: 通常Spearman r = 0.1-0.4
- **高混淆邻接对**: 通常10-50对超过阈值

## 🔗 集成使用

### 与ResNet训练结果结合
```bash
# 1. 训练ResNet获得混淆矩阵
python train_mri_resnet.py --test_subject 1

# 2. 计算对应的邻接矩阵
python compute_adjacency_matrices.py --test_subject 1

# 3. 分析两者关系
python analyze_confusion_adjacency.py --subject 1
```

### Leave-One-Out分析
```bash
# 批量处理所有被试
bash run_adjacency_batch.sh

# 与训练结果批量分析
python batch_analyze_confusion_adjacency.py
```

---

**⚠️ 注意事项**:
1. 确保3D数据与ResNet训练使用相同的文件
2. 邻接计算需要完整的`region_labels`和`region_mask`
3. 连通性选择会显著影响邻接密度
4. 建议先测试单个被试再批量处理

**📧 支持**: 如有问题请查看日志文件或运行测试脚本进行诊断。