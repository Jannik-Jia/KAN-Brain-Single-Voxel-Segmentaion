import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
import pickle
import glob
from tqdm import tqdm

class BrainVoxelDataset:
    """
    脑部体素数据集类，用于加载和处理单个数据集（训练集、验证集或测试集）
    提供标准的数据集接口，可与PyTorch等框架兼容
    """
    def __init__(self, features, labels, patient_ids=None, transform=None):
        """
        初始化数据集
        
        参数:
            features: 体素特征数据，形状为 (样本数, 特征维度)
            labels: 区域标签数据，形状为 (样本数, 标签维度)
            patient_ids: 样本对应的病人ID，形状为 (样本数,)
            transform: 特征变换函数，如标准化
        """
        assert len(features) == len(labels), "特征和标签数量必须一致"
        self.features = features
        self.labels = labels
        self.patient_ids = patient_ids
        self.transform = transform
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        feature = self.features[idx]
        label = self.labels[idx]
        
        # 应用变换（如标准化）
        if self.transform:
            feature = self.transform(feature)
        
        # 如果需要返回病人ID
        if self.patient_ids is not None:
            return feature, label, self.patient_ids[idx]
        else:
            return feature, label
    
    def get_all_data(self):
        """获取整个数据集的特征和标签"""
        features = self.features
        if self.transform:
            features = np.array([self.transform(f) for f in features])
        return features, self.labels


class BrainVoxelDataLoader:
    """
    脑部体素数据加载器，负责管理数据集划分、交叉验证和数据预处理
    """
    def __init__(self, 
                 base_dir,
                 test_patient_ids=None,
                 cross_validation=True,
                 cv_fold=30,
                 train_ratio=0.6,
                 random_seed=42,
                 standardize=True,
                 save_scaler=True,
                 output_dir='./output',
                 shuffle=True):
        """
        初始化数据加载器
        
        参数:
            base_dir: 数据根目录
            test_patient_ids: 指定测试集的病人ID，默认为[5,7,14,18,26,30,36,38]
            cross_validation: 是否启用交叉验证
            cv_fold: 交叉验证折数
            train_ratio: 非交叉验证模式下的训练集比例
            random_seed: 随机种子
            standardize: 是否进行标准化
            save_scaler: 是否保存标准化器
            output_dir: 输出目录
            shuffle: 是否打乱训练集
        """
        self.base_dir = base_dir
        self.cross_validation = cross_validation
        self.cv_fold = cv_fold
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.standardize = standardize
        self.save_scaler = save_scaler
        self.output_dir = output_dir
        self.shuffle = shuffle
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置默认测试集病人ID
        if test_patient_ids is None:
            self.test_patient_ids = [5, 7, 14, 18, 26, 30, 36, 38]
        else:
            self.test_patient_ids = test_patient_ids
        
        # 加载数据集索引
        self.dataset_index = self._load_dataset_index()
        if self.dataset_index is None:
            raise ValueError(f"无法加载数据集索引文件")
        
        # 获取所有病人ID
        self.all_patient_ids = sorted(self.dataset_index['patient_id'].unique())
        
        # 检查测试集病人ID是否有效
        for pid in self.test_patient_ids:
            if pid not in self.all_patient_ids:
                raise ValueError(f"测试集病人ID {pid} 不在数据集中")
        
        # 获取非测试集病人ID（用于训练和验证）
        self.non_test_patient_ids = sorted([pid for pid in self.all_patient_ids if pid not in self.test_patient_ids])
        
        # 检查非测试集病人数量是否足够进行交叉验证
        if cross_validation and len(self.non_test_patient_ids) < cv_fold:
            raise ValueError(f"非测试集病人数量 {len(self.non_test_patient_ids)} 小于交叉验证折数 {cv_fold}")
        
        # 初始化标准化器
        self.scaler = None
        if standardize:
            self.scaler = StandardScaler()
        
        print(f"数据加载器初始化完成")
        print(f"总病人数: {len(self.all_patient_ids)}")
        print(f"测试集病人IDs: {self.test_patient_ids}")
        print(f"训练+验证病人数: {len(self.non_test_patient_ids)}")
        if cross_validation:
            print(f"使用 {cv_fold} 折交叉验证")
        else:
            print(f"使用随机拆分: 训练集比例 {train_ratio}")
    
    def _load_dataset_index(self):
        """加载数据集索引文件"""
        index_file = os.path.join(self.base_dir, "dataset_index.csv")
        if os.path.exists(index_file):
            return pd.read_csv(index_file)
        else:
            print(f"索引文件不存在: {index_file}")
            return None
    
    def _load_patient_data(self, patient_id):
        """加载特定病人的所有区域数据"""
        patient_dir = os.path.join(self.base_dir, f"patient_{patient_id}")
        
        # 获取该病人的所有区域文件
        npy_files = glob.glob(os.path.join(patient_dir, "*.npy"))
        
        all_features = []
        all_labels = []
        all_patient_ids = []
        
        for npy_file in npy_files:
            # 从文件名提取区域ID
            filename = os.path.basename(npy_file)
            # 格式: patient_{PATIENT_ID}_region_{REGION_ID}_voxels_{COUNT}.npy
            parts = filename.split('_')
            region_id = int(parts[3])
            
            # 加载体素数据
            voxel_data = np.load(npy_file)
            
            # 为每个体素创建对应的标签（区域ID的one-hot编码）
            num_voxels = voxel_data.shape[0]
            labels = np.zeros((num_voxels, 102))  # 假设总共有102个区域
            labels[:, region_id] = 1
            
            all_features.append(voxel_data)
            all_labels.append(labels)
            all_patient_ids.append(np.full(num_voxels, patient_id))
        
        if all_features:
            # 合并所有区域的数据
            features = np.vstack(all_features)
            labels = np.vstack(all_labels)
            patient_ids = np.concatenate(all_patient_ids)
            return features, labels, patient_ids
        else:
            return None, None, None
    
    def _load_patients_data(self, patient_ids):
        """加载多个病人的数据并合并"""
        all_features = []
        all_labels = []
        all_patient_ids = []
        
        for patient_id in tqdm(patient_ids, desc="加载病人数据"):
            features, labels, p_ids = self._load_patient_data(patient_id)
            if features is not None:
                all_features.append(features)
                all_labels.append(labels)
                all_patient_ids.append(p_ids)
        
        if all_features:
            features = np.vstack(all_features)
            labels = np.vstack(all_labels)
            patient_ids = np.concatenate(all_patient_ids)
            return features, labels, patient_ids
        else:
            return None, None, None
    
    def fit_scaler(self, features):
        """使用训练数据拟合标准化器"""
        if self.standardize and self.scaler is not None:
            print("拟合标准化器...")
            self.scaler.fit(features)
            print(f"标准化器均值形状: {self.scaler.mean_.shape}")
            
            # 保存标准化器
            if self.save_scaler:
                self.save_scaler_to_file()
            
            return self.scaler
        return None
    
    def transform_data(self, features):
        """使用标准化器转换数据"""
        if self.standardize and self.scaler is not None:
            return self.scaler.transform(features)
        return features
    
    def save_scaler_to_file(self):
        """保存标准化器到文件"""
        if self.scaler is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            scaler_path = os.path.join(self.output_dir, f"scaler_{timestamp}.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"标准化器已保存到: {scaler_path}")
            return scaler_path
        return None
    
    def load_scaler_from_file(self, scaler_path):
        """从文件加载标准化器"""
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"已从 {scaler_path} 加载标准化器")
            return self.scaler
        else:
            print(f"标准化器文件不存在: {scaler_path}")
            return None
    
    def shuffle_data(self, features, labels, patient_ids=None):
        """打乱数据，保持特征和标签的对应关系"""
        if self.shuffle:
            indices = np.arange(len(features))
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
            
            features = features[indices]
            labels = labels[indices]
            if patient_ids is not None:
                patient_ids = patient_ids[indices]
            
            return features, labels, patient_ids
        return features, labels, patient_ids
    
    def get_cv_fold(self, fold_idx):
        """获取第fold_idx折交叉验证的训练集和验证集"""
        if not self.cross_validation:
            raise ValueError("交叉验证模式未启用")
        
        if fold_idx < 0 or fold_idx >= self.cv_fold:
            raise ValueError(f"fold_idx必须在0到{self.cv_fold-1}之间")
        
        # 在非测试集病人中进行k折划分
        val_patients = [self.non_test_patient_ids[fold_idx % len(self.non_test_patient_ids)]]
        train_patients = [pid for pid in self.non_test_patient_ids if pid not in val_patients]
        
        print(f"交叉验证第{fold_idx+1}/{self.cv_fold}折")
        print(f"验证集病人: {val_patients}")
        print(f"训练集病人数: {len(train_patients)}")
        
        # 加载训练集数据
        train_features, train_labels, train_patient_ids = self._load_patients_data(train_patients)
        
        # 加载验证集数据
        val_features, val_labels, val_patient_ids = self._load_patients_data(val_patients)
        
        # 打乱训练集数据
        if self.shuffle:
            train_features, train_labels, train_patient_ids = self.shuffle_data(
                train_features, train_labels, train_patient_ids)
        
        # 拟合并应用标准化
        if self.standardize:
            self.fit_scaler(train_features)
            train_transform = lambda x: self.scaler.transform(x.reshape(1, -1)).flatten() if x.ndim == 1 else self.scaler.transform(x)
            val_transform = train_transform
        else:
            train_transform = None
            val_transform = None
        
        # 创建数据集
        train_dataset = BrainVoxelDataset(train_features, train_labels, train_patient_ids, train_transform)
        val_dataset = BrainVoxelDataset(val_features, val_labels, val_patient_ids, val_transform)
        
        return train_dataset, val_dataset
    
    def get_train_val_datasets(self):
        """获取随机拆分的训练集和验证集"""
        if self.cross_validation:
            raise ValueError("交叉验证模式已启用，请使用get_cv_fold获取数据")
        
        # 加载所有非测试集病人的数据
        features, labels, patient_ids = self._load_patients_data(self.non_test_patient_ids)
        
        # 随机拆分训练集和验证集
        train_indices, val_indices = train_test_split(
            np.arange(len(features)), 
            train_size=self.train_ratio,
            random_state=self.random_seed,
            shuffle=True
        )
        
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        train_patient_ids = patient_ids[train_indices] if patient_ids is not None else None
        
        val_features = features[val_indices]
        val_labels = labels[val_indices]
        val_patient_ids = patient_ids[val_indices] if patient_ids is not None else None
        
        # 打乱训练集数据
        if self.shuffle:
            train_features, train_labels, train_patient_ids = self.shuffle_data(
                train_features, train_labels, train_patient_ids)
        
        # 拟合并应用标准化
        if self.standardize:
            self.fit_scaler(train_features)
            train_transform = lambda x: self.scaler.transform(x.reshape(1, -1)).flatten() if x.ndim == 1 else self.scaler.transform(x)
            val_transform = train_transform
        else:
            train_transform = None
            val_transform = None
        
        # 创建数据集
        train_dataset = BrainVoxelDataset(train_features, train_labels, train_patient_ids, train_transform)
        val_dataset = BrainVoxelDataset(val_features, val_labels, val_patient_ids, val_transform)
        
        print(f"随机拆分: 训练集 {len(train_dataset)} 样本，验证集 {len(val_dataset)} 样本")
        
        return train_dataset, val_dataset
    
    def get_test_dataset(self):
        """获取测试集"""
        # 加载测试集数据
        test_features, test_labels, test_patient_ids = self._load_patients_data(self.test_patient_ids)
        
        # 应用标准化（使用训练集拟合的标准化器）
        if self.standardize and self.scaler is not None:
            test_transform = lambda x: self.scaler.transform(x.reshape(1, -1)).flatten() if x.ndim == 1 else self.scaler.transform(x)
        else:
            test_transform = None
        
        # 创建数据集
        test_dataset = BrainVoxelDataset(test_features, test_labels, test_patient_ids, test_transform)
        
        print(f"测试集: {len(test_dataset)} 样本，来自 {len(self.test_patient_ids)} 个病人")
        
        return test_dataset


# 使用示例
def example_usage():
    # 初始化数据加载器（交叉验证模式）
    dataloader = BrainVoxelDataLoader(
        base_dir='./reorganized_data',
        cross_validation=True,
        standardize=True,
        output_dir='./output'
    )
    
    # 获取测试集
    test_dataset = dataloader.get_test_dataset()
    
    # 执行30折交叉验证
    for fold in range(5):  # 简化为5折用于演示
        train_dataset, val_dataset = dataloader.get_cv_fold(fold)
        
        print(f"第 {fold+1} 折:")
        print(f"  训练集大小: {len(train_dataset)}")
        print(f"  验证集大小: {len(val_dataset)}")
        
        # 这里可以使用训练集和验证集进行模型训练和评估
        # ...
    
    # 或者使用随机拆分模式
    dataloader_random = BrainVoxelDataLoader(
        base_dir='./reorganized_data',
        cross_validation=False,
        train_ratio=0.6,
        standardize=True,
        output_dir='./output'
    )
    
    train_dataset, val_dataset = dataloader_random.get_train_val_datasets()
    test_dataset = dataloader_random.get_test_dataset()
    
    print("\n随机拆分模式:")
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")


if __name__ == "__main__":
    example_usage()