# data/__init__.py 需要添加新的导入
from .dataset import BrainVoxelDataset, load_multiclass_data, apply_pca
from .samplers import BrainVoxelSampler
from .mat_loader import (
    process_train38_data, 
    create_dataloaders_from_mat, 
    load_external_mat_data, 
    BrainVoxelMatDataset,
    load_and_process_data,
    process_train38_data_with_fixed_test  # 🔧 新增
)

# 🔧 修改：统一数据加载接口需要传递config
def load_data(config, mode='train', model_path=None):
    """
    统一的数据加载接口，根据配置自动选择加载方式
    """
    # 检查配置中是否有mat_file_path，如果有则使用MAT加载方式
    if 'mat_file_path' in config and config['mat_file_path']:
        print("检测到MAT文件配置，使用MAT数据加载器")
        return load_and_process_data(config, mode, model_path)
    
    # 否则使用原始的分散文件加载方式
    elif 'data_dirs' in config and all(config['data_dirs'].values()):
        print("使用原始数据目录加载器")
        
        # 🔧 修改：需要根据背景配置调整数据加载
        dataset_dict = load_multiclass_data(
            config['data_dirs'],
            apply_pca_flag=config.get('apply_pca', False),
            n_components=config.get('n_pca', 0),
            norm=config.get('norm', True),
            config=config  # 🔧 新增：传递config参数
        )
        
        # 创建数据集
        from torch.utils.data import DataLoader
        # 创建数据集 - 传递config参数
        train_dataset = BrainVoxelDataset(dataset_dict['train_samples'], dataset_dict['train_labels'], config=config)
        test_dataset = BrainVoxelDataset(dataset_dict['test_samples'], dataset_dict['test_labels'], config=config)
        val_dataset = BrainVoxelDataset(dataset_dict['val_samples'], dataset_dict['val_labels'], config=config)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 128), shuffle=(mode=='train'))
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 128), shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 128), shuffle=False)
        
        return dataset_dict, train_loader, val_loader, test_loader
    
    else:
        raise ValueError("配置中既没有 'mat_file_path' 也没有完整的 'data_dirs'，无法加载数据")