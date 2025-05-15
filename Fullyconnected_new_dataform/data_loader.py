import numpy as np
import os
import scipy.io
import h5py
import glob
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler

class BrainVoxelDataset(Dataset):
    """脑体素数据集类，用于PyTorch数据加载"""
    def __init__(self, features, labels):
        """
        初始化数据集
        
        Args:
            features (np.ndarray): 特征数据，形状为 [n_samples, n_features]
            labels (np.ndarray): 标签数据，形状为 [n_samples]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def save_scaler(scaler, output_path, add_timestamp=True):
    """
    保存StandardScaler到指定路径，可选择添加时间戳
    
    Args:
        scaler (StandardScaler): 要保存的scaler对象
        output_path (str): 保存路径
        add_timestamp (bool): 是否添加时间戳到文件名
    
    Returns:
        str: 实际保存的文件路径
    """
    # 解析路径
    directory = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    name, ext = os.path.splitext(filename)
    
    # 确保scaler目录存在
    scaler_dir = os.path.join(directory, 'scalers')
    os.makedirs(scaler_dir, exist_ok=True)
    
    # 添加时间戳
    if add_timestamp:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name}_{timestamp}{ext}"
    else:
        new_filename = filename
    
    # 完整的保存路径
    full_save_path = os.path.join(scaler_dir, new_filename)
    
    # 保存scaler
    with open(full_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Scaler已保存到: {full_save_path}")
    
    # 同时创建一个latest链接
    latest_path = os.path.join(scaler_dir, f"{name}_latest{ext}")
    try:
        # 在Windows上可能不支持symlink，所以使用复制
        import shutil
        shutil.copy2(full_save_path, latest_path)
        print(f"同时更新了最新版本链接: {latest_path}")
    except Exception as e:
        print(f"更新最新版本链接时出错: {e}")
    
    return full_save_path

def load_scaler(scaler_path, look_for_latest=True):
    """
    从指定路径加载StandardScaler
    
    Args:
        scaler_path (str): scaler文件路径或目录
        look_for_latest (bool): 如果路径是目录，是否查找最新的scaler
        
    Returns:
        StandardScaler: 加载的scaler对象
    """
    # 检查路径是否为目录
    if os.path.isdir(scaler_path):
        if look_for_latest:
            # 查找最新的scaler
            scaler_dir = os.path.join(scaler_path, 'scalers')
            if os.path.exists(scaler_dir):
                # 获取所有.pkl文件并按修改时间排序
                scaler_files = [f for f in os.listdir(scaler_dir) if f.endswith('.pkl')]
                if not scaler_files:
                    raise FileNotFoundError(f"在{scaler_dir}中未找到.pkl文件")
                
                # 检查是否有latest文件
                latest_files = [f for f in scaler_files if '_latest.pkl' in f]
                if latest_files:
                    scaler_file = os.path.join(scaler_dir, latest_files[0])
                    print(f"使用latest scaler: {scaler_file}")
                else:
                    # 按修改时间排序
                    scaler_files.sort(key=lambda x: os.path.getmtime(os.path.join(scaler_dir, x)), reverse=True)
                    scaler_file = os.path.join(scaler_dir, scaler_files[0])
                    print(f"未找到latest scaler，使用最近修改的scaler: {scaler_file}")
                
                try:
                    with open(scaler_file, 'rb') as f:
                        scaler = pickle.load(f)
                    print(f"成功加载StandardScaler: {scaler_file}")
                    return scaler
                except Exception as e:
                    raise Exception(f"加载scaler失败: {str(e)}")
            else:
                raise FileNotFoundError(f"未找到scalers目录: {scaler_dir}")
        else:
            raise ValueError("提供的路径是目录，但look_for_latest=False")
    
    # 直接加载指定文件
    elif os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"成功加载StandardScaler: {scaler_path}")
            return scaler
        except Exception as e:
            raise Exception(f"加载scaler失败: {str(e)}")
    else:
        # 检查是否在scalers子目录中
        parent_dir = os.path.dirname(scaler_path)
        filename = os.path.basename(scaler_path)
        alt_path = os.path.join(parent_dir, 'scalers', filename)
        
        if os.path.exists(alt_path):
            try:
                with open(alt_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"成功加载StandardScaler: {alt_path}")
                return scaler
            except Exception as e:
                raise Exception(f"加载scaler失败: {str(e)}")
        
        raise FileNotFoundError(f"无法找到scaler: {scaler_path} 或 {alt_path}")

def load_region_data(directory, region_id, format='mat'):
    """
    加载单个区域的数据
    
    Args:
        directory (str): 数据目录路径
        region_id (int): 区域ID
        format (str): 数据格式，'mat'或'npy'
        
    Returns:
        dict: 包含data, region, prob_idx和可能的age的字典
    """
    result = {}
    
    if format.lower() == 'mat':
        # Mat格式加载
        file_path = os.path.join(directory, 'mat', f"region_{region_id}.mat")
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        try:
            # 尝试使用scipy.io.loadmat加载
            mat_data = scipy.io.loadmat(file_path)
            for key in mat_data:
                if key in ['data', 'region', 'prob_idx', 'age']:
                    result[key] = mat_data[key]
        except:
            # 如果失败，尝试使用h5py加载
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    if key in ['data', 'region', 'prob_idx', 'age']:
                        result[key] = np.array(f[key])
                        # 如果需要转置
                        if result[key].ndim > 1:
                            result[key] = result[key].transpose()
    
    elif format.lower() == 'npy':
        # Npy格式加载
        npy_dir = os.path.join(directory, 'npy')
        
        # 加载数据
        data_path = os.path.join(npy_dir, f"region_{region_id}_data.npy")
        if os.path.exists(data_path):
            result['data'] = np.load(data_path)
        else:
            print(f"文件不存在: {data_path}")
            return None
        
        # 加载区域标签
        region_path = os.path.join(npy_dir, f"region_{region_id}_region.npy")
        if os.path.exists(region_path):
            result['region'] = np.load(region_path)
        
        # 加载病人ID
        prob_idx_path = os.path.join(npy_dir, f"region_{region_id}_prob_idx.npy")
        if os.path.exists(prob_idx_path):
            result['prob_idx'] = np.load(prob_idx_path)
        
        # 加载年龄数据（如果有）
        age_path = os.path.join(npy_dir, f"region_{region_id}_age.npy")
        if os.path.exists(age_path):
            result['age'] = np.load(age_path)
    
    else:
        print(f"不支持的格式: {format}")
        return None
    
    return result

def get_active_regions(directory, format='mat'):
    """
    获取目录中的所有活跃区域ID
    
    Args:
        directory (str): 数据目录路径
        format (str): 数据格式，'mat'或'npy'
        
    Returns:
        list: 活跃区域ID列表
    """
    active_regions = []
    
    if format.lower() == 'mat':
        # Mat格式
        mat_dir = os.path.join(directory, 'mat')
        if os.path.exists(mat_dir):
            files = glob.glob(os.path.join(mat_dir, "region_*.mat"))
            for file in files:
                try:
                    region_id = int(os.path.basename(file).split('_')[1].split('.')[0])
                    active_regions.append(region_id)
                except:
                    pass
    
    elif format.lower() == 'npy':
        # Npy格式
        npy_dir = os.path.join(directory, 'npy')
        if os.path.exists(npy_dir):
            files = glob.glob(os.path.join(npy_dir, "region_*_data.npy"))
            for file in files:
                try:
                    region_id = int(os.path.basename(file).split('_')[1])
                    active_regions.append(region_id)
                except:
                    pass
    
    return sorted(active_regions)

def load_all_regions(directory, region_ids=None, format='mat'):
    """
    加载指定目录下的所有区域数据或指定区域数据
    
    Args:
        directory (str): 数据目录路径
        region_ids (list): 要加载的区域ID列表，如果为None则加载所有区域
        format (str): 数据格式，'mat'或'npy'
        
    Returns:
        dict: 包含所有区域数据的字典，键为区域ID
    """
    if region_ids is None:
        region_ids = get_active_regions(directory, format)
    
    if not region_ids:
        print(f"未找到活跃区域，请检查目录: {directory}")
        return {}
    
    result = {}
    for region_id in tqdm(region_ids, desc=f"加载区域数据"):
        region_data = load_region_data(directory, region_id, format)
        if region_data is not None:
            result[region_id] = region_data
    
    return result

def load_brain_voxel_data(base_dir, split='train', format='mat', shuffle=True, seed=666):
    """
    从目录中加载脑体素数据
    
    Args:
        base_dir (str): 数据基础目录路径
        split (str): 数据集划分，'train', 'val'或'test'
        format (str): 数据格式，'mat'或'npy'
        shuffle (bool): 是否打乱数据
        seed (int): 随机种子
        
    Returns:
        dict: 包含样本和标签的数据字典
    """
    print(f"从 {base_dir} 加载 {split} 数据集...")
    
    # 设置目标目录
    target_dir = os.path.join(base_dir, split)
    
    # 加载所有区域数据
    print(f"加载所有区域数据...")
    regions_data = load_all_regions(target_dir, format=format)
    
    if not regions_data:
        raise ValueError(f"未找到区域数据，请检查目录: {target_dir}")
    
    # 初始化存储所有样本和标签的列表
    all_features = []
    all_labels = []
    
    print(f"处理区域数据...")
    for region_id, region_data in tqdm(regions_data.items(), desc=f"处理{split}区域"):
        if 'data' not in region_data or 'region' not in region_data:
            print(f"警告: 区域 {region_id} 缺少必要的数据字段，跳过")
            continue
        
        # 获取该区域的特征数据
        features = region_data['data']
        
        # 获取该区域的标签数据
        region_labels = region_data['region']
        
        # 确定每个样本的类别索引
        # 检查region_labels的形状确定是否为one-hot
        if len(region_labels.shape) > 1 and region_labels.shape[1] > 1:  # 如果是one-hot编码
            sample_labels = np.argmax(region_labels, axis=1)
        else:  # 如果已经是类别索引
            sample_labels = region_labels.flatten()
        
        # 添加到总列表
        all_features.append(features)
        all_labels.append(sample_labels)
    
    # 合并所有区域的数据
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    print(f"{split} 数据集总样本数: {len(all_features)}")
    
    # 全局打乱数据（如果需要）
    if shuffle:
        print(f"全局打乱 {split} 数据集...")
        np.random.seed(seed)
        indices = np.random.permutation(len(all_features))
        all_features = all_features[indices]
        all_labels = all_labels[indices]
        print(f"完成全局打乱，确保特征和标签的对应关系")
    
    return {
        'features': all_features,
        'labels': all_labels,
        'feature_dim': all_features.shape[1],
        'num_classes': len(np.unique(all_labels))
    }

def create_data_loaders(dataset_dict, batch_size=128, shuffle_train=True):
    """
    从数据字典创建PyTorch数据加载器
    
    Args:
        dataset_dict (dict): 包含features和labels的数据字典
        batch_size (int): 批次大小
        shuffle_train (bool): 是否打乱训练数据
        
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = BrainVoxelDataset(
        dataset_dict['features'],
        dataset_dict['labels']
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle_train
    )

def check_standardization(data, threshold=0.1):
    """
    检查数据是否已标准化
    
    Args:
        data: numpy数组或PyTorch张量，形状为 [n_samples, n_features]
        threshold: 均值和标准差允许的偏差阈值
        
    Returns:
        bool: 数据是否已标准化
    """
    # 如果是PyTorch张量，转换为numpy数组
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # 计算每个特征的均值和标准差
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # 打印统计信息
    print(f"特征均值范围: [{means.min():.4f}, {means.max():.4f}], 平均={np.mean(means):.4f}")
    print(f"特征标准差范围: [{stds.min():.4f}, {stds.max():.4f}], 平均={np.mean(stds):.4f}")
    
    # 检查均值是否接近0，标准差是否接近1
    mean_close_to_zero = np.all(np.abs(means) < threshold)
    std_close_to_one = np.all(np.abs(stds - 1.0) < threshold)
    
    return mean_close_to_zero and std_close_to_one

def load_and_prepare_data(base_dir, scaler_path=None, create_new_scaler=True, format='mat', 
                          batch_size=128, shuffle=True, seed=666):
    """
    加载数据并创建数据加载器，支持标准化处理
    
    Args:
        base_dir (str): 数据基础目录路径
        scaler_path (str): scaler文件路径或目录，如果为None则使用默认路径
        create_new_scaler (bool): 是否创建新的scaler，如果为False则尝试加载现有scaler
        format (str): 数据格式，'mat'或'npy'
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        seed (int): 随机种子
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, feature_dim, num_classes, scaler)
    """
    # 设置默认scaler路径
    if scaler_path is None:
        scaler_path = base_dir
    
    # 加载scaler或创建新的scaler
    scaler = None
    if not create_new_scaler:
        # 尝试加载现有scaler
        print(f"\n{'='*80}")
        print(f"正在尝试加载现有scaler...")
        print(f"{'='*80}")
        try:
            scaler = load_scaler(scaler_path)
            print(f"成功加载StandardScaler，特征数量: {len(scaler.mean_)}")
        except FileNotFoundError as e:
            print(f"{e}，将创建新的scaler")
            create_new_scaler = True
        except Exception as e:
            print(f"加载scaler时出错: {str(e)}，将创建新的scaler")
            create_new_scaler = True
    
    # 加载训练集
    train_data = load_brain_voxel_data(
        base_dir=base_dir, 
        split='train', 
        format=format,
        shuffle=shuffle,
        seed=seed
    )
    
    # 加载验证集
    val_data = load_brain_voxel_data(
        base_dir=base_dir, 
        split='val', 
        format=format,
        shuffle=shuffle,
        seed=seed
    )
    
    # 加载测试集
    test_data = load_brain_voxel_data(
        base_dir=base_dir, 
        split='test', 
        format=format,
        shuffle=shuffle,
        seed=seed
    )
    
    # 如果需要创建新的scaler
    if create_new_scaler:
        print(f"\n{'='*80}")
        print(f"注意: 正在创建新的scaler!")
        print(f"新的scaler将基于当前训练集数据的分布进行拟合")
        print(f"这可能会影响模型性能，特别是在迁移到新数据时")
        print(f"{'='*80}\n")
        
        print("使用训练集创建新的scaler...")
        scaler = StandardScaler()
        scaler.fit(train_data['features'])
        
        # 构造scaler保存名称
        scaler_name = f"brain_voxel_scaler.pkl"
        
        # 保存新的scaler，添加时间戳
        saved_path = save_scaler(scaler, os.path.join(scaler_path, scaler_name), add_timestamp=True)
        print(f"\n{'*'*80}")
        print(f"新的scaler已创建并保存到: {saved_path}")
        print(f"{'*'*80}\n")
    
    # 应用标准化
    print("对所有数据集应用标准化...")
    train_data['features'] = scaler.transform(train_data['features'])
    val_data['features'] = scaler.transform(val_data['features'])
    test_data['features'] = scaler.transform(test_data['features'])
    
    # 检查标准化效果
    print("\n检查训练集标准化效果:")
    train_standardized = check_standardization(train_data['features'])
    print(f"训练集标准化状态: {train_standardized}")
    
    # 创建数据加载器
    train_loader = create_data_loaders(train_data, batch_size, shuffle_train=True)
    val_loader = create_data_loaders(val_data, batch_size, shuffle_train=False)
    test_loader = create_data_loaders(test_data, batch_size, shuffle_train=False)
    
    return train_loader, val_loader, test_loader, train_data['feature_dim'], train_data['num_classes'], scaler

def analyze_data_distribution(train_loader, val_loader, test_loader, sample_size=1000):
    """
    分析数据集的分布情况
    
    Args:
        train_loader (DataLoader): 训练集数据加载器
        val_loader (DataLoader): 验证集数据加载器
        test_loader (DataLoader): 测试集数据加载器
        sample_size (int): 用于检查重叠的样本数量
        
    Returns:
        dict: 分布分析结果
    """
    print("提取数据样本进行分析...")
    
    # 提取样本
    def extract_samples(loader, n=sample_size):
        features = []
        labels = []
        for batch_features, batch_labels in loader:
            features.append(batch_features[:min(len(batch_features), n - len(features))])
            labels.append(batch_labels[:min(len(batch_labels), n - len(features))])
            if len(features) * features[0].shape[0] >= n:
                break
        return torch.cat(features, 0).numpy(), torch.cat(labels, 0).numpy()
    
    train_features, train_labels = extract_samples(train_loader)
    val_features, val_labels = extract_samples(val_loader)
    test_features, test_labels = extract_samples(test_loader)
    
    # 检查数据集之间的重叠
    print("验证数据集分割独立性（检查是否有重叠）...")
    
    def count_overlaps(features1, features2):
        # 计算两个数据集之间的重叠数量（近似）
        threshold = 1e-6  # 相似度阈值
        overlaps = 0
        
        for i in range(min(len(features1), sample_size)):
            for j in range(min(len(features2), sample_size)):
                if np.all(np.abs(features1[i] - features2[j]) < threshold):
                    overlaps += 1
                    break
        
        return overlaps
    
    train_val_overlaps = count_overlaps(train_features, val_features)
    train_test_overlaps = count_overlaps(train_features, test_features)
    val_test_overlaps = count_overlaps(val_features, test_features)
    
    print(f"训练集与验证集重叠样本数: {train_val_overlaps} / {sample_size}")
    print(f"训练集与测试集重叠样本数: {train_test_overlaps} / {sample_size}")
    print(f"验证集与测试集重叠样本数: {val_test_overlaps} / {sample_size}")
    
    if max(train_val_overlaps, train_test_overlaps, val_test_overlaps) < sample_size * 0.01:
        print("数据集分割正确：训练集、验证集和测试集之间没有显著重叠。")
    else:
        print("警告：数据集之间存在显著重叠！这可能会导致模型评估不准确。")
    
    # 检查标签分布
    print("\n检查标签分布...")
    
    def get_label_distribution(labels):
        unique_labels = np.unique(labels)
        distribution = {}
        for label in unique_labels:
            distribution[label] = np.sum(labels == label) / len(labels) * 100
        return distribution
    
    train_dist = get_label_distribution(train_labels)
    val_dist = get_label_distribution(val_labels)
    test_dist = get_label_distribution(test_labels)
    
    all_labels = set(list(train_dist.keys()) + list(val_dist.keys()) + list(test_dist.keys()))
    
    print(f"数据集中的类别总数: {len(all_labels)}")
    
    print("\n各数据集中主要类别的分布比例（前10个类别）:")
    print(f"{'类别ID':<10}{'训练集比例':<20}{'验证集比例':<20}{'测试集比例':<20}")
    print("-" * 55)
    
    for i, label in enumerate(sorted(list(all_labels))[:10]):
        train_pct = train_dist.get(label, 0)
        val_pct = val_dist.get(label, 0)
        test_pct = test_dist.get(label, 0)
        print(f"{int(label):<10}{train_pct:<20.2f}{val_pct:<20.2f}{test_pct:<20.2f}")
    
    # 计算分布差异
    def distribution_difference(dist1, dist2):
        all_keys = set(list(dist1.keys()) + list(dist2.keys()))
        diff = 0
        for key in all_keys:
            diff += abs(dist1.get(key, 0) - dist2.get(key, 0))
        return diff / len(all_keys) / 100  # 归一化
    
    train_val_diff = distribution_difference(train_dist, val_dist)
    train_test_diff = distribution_difference(train_dist, test_dist)
    val_test_diff = distribution_difference(val_dist, test_dist)
    
    print("\n分布差异度量（越小表示分布越相似）:")
    print(f"训练集-验证集差异: {train_val_diff:.4f}")
    print(f"训练集-测试集差异: {train_test_diff:.4f}")
    print(f"验证集-测试集差异: {val_test_diff:.4f}")
    
    if max(train_val_diff, train_test_diff, val_test_diff) < 0.1:
        print("\n结论: 所有数据集具有相似的标签分布，这表明数据分割良好。")
    else:
        print("\n警告: 数据集之间的标签分布存在明显差异，这可能会影响模型性能。")
    
    # 数据集大小信息
    def get_dataset_size(loader):
        return loader.dataset.__len__()
    
    train_size = get_dataset_size(train_loader)
    val_size = get_dataset_size(val_loader)
    test_size = get_dataset_size(test_loader)
    total_size = train_size + val_size + test_size
    
    print("\n数据集大小信息:")
    print(f"训练集: {train_size:,} 样本")
    print(f"验证集: {val_size:,} 样本")
    print(f"测试集: {test_size:,} 样本")
    print(f"总样本数: {total_size:,}")
    
    print("\n数据集比例:")
    train_pct = train_size / total_size * 100
    val_pct = val_size / total_size * 100
    test_pct = test_size / total_size * 100
    
    print(f"训练集: {train_pct:.2f}%")
    print(f"验证集: {val_pct:.2f}%")
    print(f"测试集: {test_pct:.2f}%")
    
    # 检查数据集比例是否在常规范围内
    if 60 <= train_pct <= 85 and 5 <= val_pct <= 20 and 5 <= test_pct <= 20:
        print("数据集比例在正常范围内。")
    else:
        print("数据集比例不在典型范围内，可能需要检查分割逻辑。")
    
    return {
        "overlap_analysis": {
            "train_val_overlaps": train_val_overlaps,
            "train_test_overlaps": train_test_overlaps,
            "val_test_overlaps": val_test_overlaps
        },
        "distribution_analysis": {
            "train_dist": train_dist,
            "val_dist": val_dist,
            "test_dist": test_dist,
            "train_val_diff": train_val_diff,
            "train_test_diff": train_test_diff,
            "val_test_diff": val_test_diff
        },
        "size_analysis": {
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "total_size": total_size,
            "train_pct": train_pct,
            "val_pct": val_pct,
            "test_pct": test_pct
        }
    }

if __name__ == "__main__":
    # 测试代码
    base_dir = "/path/to/data"
    
    print("\n" + "="*100)
    print("脑体素数据加载器使用说明")
    print("="*100)
    print("\n此数据加载器提供了以下主要功能:")
    print("1. 加载和准备脑体素数据，包括训练集、验证集和测试集")
    print("2. 创建或加载数据标准化器(scaler)，支持时间戳版本控制")
    print("3. 对数据进行标准化处理")
    print("4. 分析数据集分布")
    
    print("\n" + "-"*100)
    print("用法示例:")
    print("-"*100)
    
    print("\n示例1: 创建新的scaler并应用")
    print("```python")
    print("from data_loader import load_and_prepare_data, analyze_data_distribution")
    print("")
    print("# 创建新的scaler并应用于数据")
    print("train_loader, val_loader, test_loader, feature_dim, num_classes, scaler = load_and_prepare_data(")
    print("    base_dir='/path/to/data',")
    print("    create_new_scaler=True,  # 关键参数：创建新的scaler")
    print("    batch_size=128,")
    print("    shuffle=True,")
    print("    seed=666")
    print(")")
    print("```")
    print("说明: 此模式会使用当前训练集创建一个新的scaler，并自动保存到'/path/to/data/scalers/'目录下，")
    print("     文件名格式为'brain_voxel_scaler_YYYYMMDD_HHMMSS.pkl'，同时更新latest版本。")
    
    print("\n示例2: 加载最新的scaler并应用")
    print("```python")
    print("# 加载现有的最新scaler并应用于数据")
    print("train_loader, val_loader, test_loader, feature_dim, num_classes, scaler = load_and_prepare_data(")
    print("    base_dir='/path/to/data',")
    print("    create_new_scaler=False,  # 关键参数：加载现有scaler")
    print("    batch_size=128")
    print(")")
    print("```")
    print("说明: 此模式会自动在'/path/to/data/scalers/'目录下查找最新的scaler或带有'latest'标记的scaler。")
    
    print("\n示例3: 加载指定路径的scaler")
    print("```python")
    print("# 加载指定路径的scaler并应用于数据")
    print("specific_scaler = '/path/to/specific/scalers/brain_voxel_scaler_20240515_120000.pkl'")
    print("train_loader, val_loader, test_loader, feature_dim, num_classes, scaler = load_and_prepare_data(")
    print("    base_dir='/path/to/data',")
    print("    create_new_scaler=False,  # 加载现有scaler")
    print("    scaler_path=specific_scaler,  # 指定具体的scaler文件")
    print("    batch_size=128")
    print(")")
    print("```")
    print("说明: 此模式会加载指定路径的特定scaler，适用于需要使用历史版本scaler的情况。")
    
    print("\n示例4: 数据分布分析")
    print("```python")
    print("# 分析数据集分布情况")
    print("analysis_results = analyze_data_distribution(train_loader, val_loader, test_loader)")
    print("```")
    print("说明: 此函数会分析数据集的分布情况，包括标签分布、数据集重叠情况和数据集大小比例。")
    
    print("\n" + "-"*100)
    print("注意事项:")
    print("-"*100)
    print("1. 创建新的scaler会影响模型的性能，特别是在数据分布变化时")
    print("2. 对于生产环境，建议使用固定的scaler以确保一致性")
    print("3. scaler目录会保存历史版本，方便追踪和回溯")
    print("4. 在迁移学习或泛化到新数据集时，可能需要重新创建scaler")
    print("\n" + "="*100)
    
    # 实际测试代码可以在这里运行
    # 示例1：创建新的scaler并应用
    """
    train_loader, val_loader, test_loader, feature_dim, num_classes, scaler = load_and_prepare_data(
        base_dir=base_dir, 
        create_new_scaler=True,  # 创建新的scaler
        # scaler自动保存到 base_dir/scalers/brain_voxel_scaler_YYYYMMDD_HHMMSS.pkl
        format='mat',
        batch_size=128,
        shuffle=True,
        seed=666
    )
    
    # 示例2：加载现有scaler并应用
    train_loader, val_loader, test_loader, feature_dim, num_classes, scaler = load_and_prepare_data(
        base_dir=base_dir, 
        create_new_scaler=False,  # 加载现有scaler
        # 会自动查找 base_dir/scalers/ 下的最新scaler或使用latest版本
        format='mat',
        batch_size=128,
        shuffle=True,
        seed=666
    )
    
    # 示例3：加载指定路径的scaler
    specific_scaler = "/path/to/specific/scaler.pkl"
    train_loader, val_loader, test_loader, feature_dim, num_classes, scaler = load_and_prepare_data(
        base_dir=base_dir, 
        create_new_scaler=False,  # 加载现有scaler
        scaler_path=specific_scaler,  # 指定具体的scaler文件
        format='mat',
        batch_size=128,
        shuffle=True,
        seed=666
    )
    
    # 分析数据分布
    analysis_results = analyze_data_distribution(train_loader, val_loader, test_loader)
    """