# 3D脑体素分类网络架构设计指南

## 项目概述

本文档提供了基于3×3和3×3×3卷积的3D脑体素分类网络的完整架构设计。针对384×336×256×351维的多模态MRI数据，设计了从1D全连接网络到3D卷积网络的升级方案，结合先进的体素采样策略和网络优化技术。

### 数据集特性
- **3D体积尺寸**: 384 × 336 × 256 体素
- **特征维度**: 351维多模态MRI特征（T1加权、CEST、QSM等）
- **目标类别**: 102个脑区域（FreeSurfer分割）
- **有效体素**: 约150万-240万个（每个被试）
- **被试数量**: 38个

## 网络架构演进路径

### 1. 从1D全连接到3D卷积的架构演进

#### A) 原始1D全连接网络（基线）

```python
class FCNet_1D(nn.Module):
    """原始1D全连接网络：单体素分类"""
    
    def __init__(self, input_dim=351, num_classes=102):
        super(FCNet_1D, self).__init__()
        
        # 4层全连接网络
        self.fc1 = nn.Linear(351, 4096)      # 351 -> 4096
        self.fc2 = nn.Linear(4096, 4096)     # 4096 -> 4096  
        self.fc3 = nn.Linear(4096, 4096)     # 4096 -> 4096
        self.fc4 = nn.Linear(4096, 102)      # 4096 -> 102类
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch_size, 351) - 单个体素的351维特征
        x = F.relu(self.fc1(x))          # (batch_size, 4096)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))          # (batch_size, 4096)
        x = self.dropout(x)
        x = F.relu(self.fc3(x))          # (batch_size, 4096)
        x = self.dropout(x)
        x = self.fc4(x)                  # (batch_size, 102)
        return x

# 特点：
# - 参数量：~67M参数
# - 输入：单体素351维特征
# - 学习内容：纯特征变换
# - 缺点：无空间上下文信息
```

#### B) 2D Patch卷积网络（中级）

```python
class Conv2D_Patch_Net(nn.Module):
    """2D Patch网络：学习空间邻域关系"""
    
    def __init__(self, input_channels=351, num_classes=102, patch_size=3):
        super(Conv2D_Patch_Net, self).__init__()
        
        self.patch_size = patch_size
        
        # 2D卷积层：处理空间邻域
        self.conv2d_1 = nn.Conv2d(
            in_channels=351,     # 351个MRI特征通道
            out_channels=128,    # 输出128个特征图
            kernel_size=3,       # 3×3卷积核
            padding=0           # 无padding：3×3 -> 1×1
        )
        
        self.conv2d_2 = nn.Conv2d(128, 256, kernel_size=1)  # 1×1卷积特征增强
        self.conv2d_3 = nn.Conv2d(256, 512, kernel_size=1)  # 进一步特征提取
        
        # 批归一化和激活
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        
        # 分类器：相比1D网络参数更少
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 351, 3, 3) - 3×3 patch
        
        x = F.relu(self.bn1(self.conv2d_1(x)))  # (batch_size, 128, 1, 1)
        x = F.relu(self.bn2(self.conv2d_2(x)))  # (batch_size, 256, 1, 1)  
        x = F.relu(self.bn3(self.conv2d_3(x)))  # (batch_size, 512, 1, 1)
        
        # 展平进入分类器
        x = x.view(x.size(0), -1)               # (batch_size, 512)
        x = self.classifier(x)                  # (batch_size, 102)
        
        return x

# 特点：
# - 参数量：~5M参数（显著减少）
# - 输入：3×3×351维patch
# - 学习内容：2D空间邻域+特征关系
# - 优势：捕获平面内的解剖连续性
```

#### C) 3D体积卷积网络（高级）

```python
class Conv3D_Voxel_Net(nn.Module):
    """3D体积网络：完整的3D空间上下文学习"""
    
    def __init__(self, input_channels=351, num_classes=102, patch_size=3):
        super(Conv3D_Voxel_Net, self).__init__()
        
        self.patch_size = patch_size
        
        # 3D卷积层：学习完整3D空间结构
        self.conv3d_1 = nn.Conv3d(
            in_channels=351,     # 351个MRI特征通道
            out_channels=128,    # 输出128个3D特征图
            kernel_size=3,       # 3×3×3卷积核
            padding=0           # 3×3×3 -> 1×1×1
        )
        
        # 特征增强层
        self.conv3d_2 = nn.Conv3d(128, 256, kernel_size=1)  # 1×1×1卷积
        self.conv3d_3 = nn.Conv3d(256, 512, kernel_size=1)  
        
        # 3D批归一化
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(256) 
        self.bn3 = nn.BatchNorm3d(512)
        
        # 3D dropout
        self.dropout3d = nn.Dropout3d(0.3)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 351, 3, 3, 3) - 3×3×3体素patch
        
        x = F.relu(self.bn1(self.conv3d_1(x)))  # (batch_size, 128, 1, 1, 1)
        x = self.dropout3d(x)
        
        x = F.relu(self.bn2(self.conv3d_2(x)))  # (batch_size, 256, 1, 1, 1)
        x = self.dropout3d(x)
        
        x = F.relu(self.bn3(self.conv3d_3(x)))  # (batch_size, 512, 1, 1, 1)
        
        # 展平进入分类器
        x = x.view(x.size(0), -1)               # (batch_size, 512)
        x = self.classifier(x)                  # (batch_size, 102)
        
        return x

# 特点：
# - 参数量：~6M参数
# - 输入：3×3×3×351维体素块
# - 学习内容：完整3D空间邻域关系
# - 优势：捕获3D解剖连续性和层次结构
```

## 高级体素采样策略

### 1. 基础Patch采样（借鉴KAN方法）

```python
class Brain3DPatchSet(Dataset):
    """3D脑体素patch采样器 - 基于HSIConvKAN采样策略"""
    
    def __init__(self, volume_4d, labels_3d, region_mask, patch_size=3, is_pred=False):
        """
        Args:
            volume_4d: (384, 336, 256, 351) - 4D特征体积
            labels_3d: (384, 336, 256) - 标签体积  
            region_mask: (384, 336, 256) - 有效区域掩膜
            patch_size: int - patch尺寸（3表示3×3×3）
            is_pred: bool - 预测模式
        """
        super(Brain3DPatchSet, self).__init__()
        
        self.is_pred = is_pred
        self.patch_size = patch_size
        self.volume_4d = volume_4d
        self.labels_3d = labels_3d
        self.region_mask = region_mask
        
        # 计算填充大小（类似KAN的边界处理）
        p = self.patch_size // 2
        
        # 3D零填充：在三个空间维度上填充，特征维度不变
        self.padded_volume = np.pad(
            volume_4d, 
            ((p,p), (p,p), (p,p), (0,0)), 
            'constant', 
            constant_values=0
        )
        
        # 标签和掩膜也需要填充
        self.padded_labels = np.pad(labels_3d, (p,p), 'constant', constant_values=0)
        self.padded_mask = np.pad(region_mask, (p,p), 'constant', constant_values=0)
        
        # 获取有效体素位置
        if is_pred:
            # 预测模式：处理所有有效体素
            valid_coords = np.where(region_mask == 1)
        else:
            # 训练模式：只处理有标签的体素
            valid_coords = np.where((region_mask == 1) & (labels_3d > 0))
        
        # 调整坐标以补偿填充（关键步骤，借鉴KAN）
        self.indices = []
        for i in range(len(valid_coords[0])):
            x, y, z = valid_coords[0][i] + p, valid_coords[1][i] + p, valid_coords[2][i] + p
            self.indices.append((x, y, z))
        
        # 训练时随机打乱
        if not is_pred:
            np.random.shuffle(self.indices)

    def __getitem__(self, i):
        x, y, z = self.indices[i]
        
        # 计算3D patch边界
        p = self.patch_size // 2
        x1, x2 = x - p, x + p + 1
        y1, y2 = y - p, y + p + 1  
        z1, z2 = z - p, z + p + 1
        
        # 提取3×3×3×351的patch
        patch = self.padded_volume[x1:x2, y1:y2, z1:z2, :]  # (3, 3, 3, 351)
        
        # 获取中心体素的标签
        label = self.padded_labels[x, y, z]
        
        # 转换为PyTorch格式：(C, D, H, W)
        patch = patch.transpose(3, 0, 1, 2).astype(np.float32)  # (351, 3, 3, 3)
        
        patch_tensor = torch.from_numpy(patch)
        label_tensor = torch.from_numpy(np.array(label, dtype=np.int64))
        
        if self.is_pred:
            return patch_tensor
        else:
            return patch_tensor, label_tensor

    def __len__(self):
        return len(self.indices)
```

### 2. 高级采样策略

#### A) 多尺度patch采样

```python
class MultiScaleBrain3DPatch(Dataset):
    """多尺度3D patch采样：提供不同感受野的上下文"""
    
    def __init__(self, volume_4d, labels_3d, region_mask, patch_sizes=[3, 5, 7]):
        self.patch_sizes = patch_sizes
        self.patch_datasets = []
        
        # 为每个尺度创建采样器
        for patch_size in patch_sizes:
            dataset = Brain3DPatchSet(volume_4d, labels_3d, region_mask, patch_size)
            self.patch_datasets.append(dataset)
    
    def __getitem__(self, i):
        patches = []
        label = None
        
        for dataset in self.patch_datasets:
            patch, lbl = dataset[i]
            patches.append(patch)
            if label is None:
                label = lbl
        
        # 在通道维度拼接多尺度特征
        multi_scale_patch = torch.cat(patches, dim=0)  # (351*3, max_patch_size, ...)
        
        return multi_scale_patch, label

class MultiScaleConv3DNet(nn.Module):
    """处理多尺度patch的网络"""
    
    def __init__(self, num_classes=102):
        super().__init__()
        
        # 分别处理不同尺度
        self.scale_processors = nn.ModuleList([
            nn.Conv3d(351, 128, kernel_size=3, padding=0),  # 3×3×3
            nn.Conv3d(351, 128, kernel_size=5, padding=0),  # 5×5×5  
            nn.Conv3d(351, 128, kernel_size=7, padding=0),  # 7×7×7
        ])
        
        # 特征融合
        self.fusion_conv = nn.Conv3d(384, 512, kernel_size=1)  # 128*3=384
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # x: multi-scale patches
        scale_features = []
        
        start_idx = 0
        for i, processor in enumerate(self.scale_processors):
            end_idx = start_idx + 351
            scale_input = x[:, start_idx:end_idx, ...]
            scale_feature = F.relu(processor(scale_input))
            scale_features.append(scale_feature)
            start_idx = end_idx
        
        # 融合多尺度特征
        fused = torch.cat(scale_features, dim=1)  # 拼接通道
        fused = F.relu(self.fusion_conv(fused))
        
        # 分类
        fused = fused.view(fused.size(0), -1)
        output = self.classifier(fused)
        
        return output
```

#### B) 自适应采样策略

```python
def adaptive_patch_sampling(volume_4d, labels_3d, region_mask, base_size=3):
    """根据局部复杂度自适应调整patch大小"""
    
    adaptive_sizes = {}
    valid_coords = np.where(region_mask == 1)
    
    for i in range(len(valid_coords[0])):
        x, y, z = valid_coords[0][i], valid_coords[1][i], valid_coords[2][i]
        
        # 计算局部方差作为复杂度指标
        local_region = volume_4d[max(0,x-2):min(384,x+3), 
                                max(0,y-2):min(336,y+3), 
                                max(0,z-2):min(256,z+3), :]
        
        # 跨通道计算方差
        local_variance = np.var(local_region)
        
        # 根据复杂度调整patch大小
        if local_variance > 0.8:  # 高复杂度区域
            patch_size = base_size + 2  # 使用更大patch
        elif local_variance < 0.2:  # 低复杂度区域
            patch_size = base_size      # 标准patch
        else:
            patch_size = base_size + 1  # 中等patch
            
        adaptive_sizes[(x, y, z)] = patch_size
    
    return adaptive_sizes

class AdaptiveBrain3DNet(nn.Module):
    """自适应patch大小的网络"""
    
    def __init__(self, num_classes=102):
        super().__init__()
        
        # 不同patch size的处理器
        self.processors = nn.ModuleDict({
            '3': Conv3D_Voxel_Net(351, num_classes, 3),
            '5': Conv3D_Voxel_Net(351, num_classes, 5), 
            '7': Conv3D_Voxel_Net(351, num_classes, 7),
        })
    
    def forward(self, x, patch_size):
        processor = self.processors[str(patch_size)]
        return processor(x)
```

### 3. 边界和特殊区域处理

```python
class BoundaryAwareSampling:
    """边界感知的采样策略"""
    
    @staticmethod
    def is_boundary_voxel(x, y, z, region_mask):
        """检查体素是否在区域边界上"""
        if (x <= 0 or x >= region_mask.shape[0]-1 or
            y <= 0 or y >= region_mask.shape[1]-1 or  
            z <= 0 or z >= region_mask.shape[2]-1):
            return True
            
        # 检查6邻域
        neighbors = [
            region_mask[x-1,y,z], region_mask[x+1,y,z],
            region_mask[x,y-1,z], region_mask[x,y+1,z],
            region_mask[x,y,z-1], region_mask[x,y,z+1]
        ]
        
        # 如果存在背景邻居，则为边界
        return 0 in neighbors
    
    @staticmethod
    def enhanced_boundary_sampling(volume_4d, labels_3d, region_mask, 
                                 boundary_ratio=2.0):
        """增强边界区域的采样"""
        
        all_indices = []
        valid_coords = np.where((region_mask == 1) & (labels_3d > 0))
        
        boundary_indices = []
        interior_indices = []
        
        for i in range(len(valid_coords[0])):
            x, y, z = valid_coords[0][i], valid_coords[1][i], valid_coords[2][i]
            
            if BoundaryAwareSampling.is_boundary_voxel(x, y, z, region_mask):
                boundary_indices.append((x, y, z))
            else:
                interior_indices.append((x, y, z))
        
        # 按比例采样
        n_boundary = len(boundary_indices)
        n_interior = int(n_boundary / boundary_ratio)  # 减少内部采样
        
        # 随机选择内部点
        if n_interior < len(interior_indices):
            interior_indices = np.random.choice(
                len(interior_indices), n_interior, replace=False
            )
            interior_indices = [interior_indices[i] for i in interior_indices]
        
        return boundary_indices + interior_indices
```

## 完整训练框架

### 1. 数据加载器

```python
class BrainVolumeDataLoader:
    """完整的3D脑体积数据加载系统"""
    
    def __init__(self, data_dir, patch_size=3, batch_size=256, num_workers=4):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def create_datasets(self, train_ratio=0.7, val_ratio=0.15):
        """创建训练、验证、测试数据集"""
        
        # 加载所有被试数据
        all_volumes = []
        all_labels = []
        all_masks = []
        
        for subject_file in glob.glob(f"{self.data_dir}/*.mat"):
            data = scipy.io.loadmat(subject_file)
            
            volume = data['data']           # (384, 336, 256, 351)
            labels = data['region_labels']  # (384, 336, 256)
            mask = data['region_mask']      # (384, 336, 256)
            
            all_volumes.append(volume)
            all_labels.append(labels)
            all_masks.append(mask)
        
        # 创建patch数据集
        all_patches = []
        all_patch_labels = []
        
        for volume, labels, mask in zip(all_volumes, all_labels, all_masks):
            patch_dataset = Brain3DPatchSet(volume, labels, mask, self.patch_size)
            
            for i in range(len(patch_dataset)):
                patch, label = patch_dataset[i]
                all_patches.append(patch)
                all_patch_labels.append(label)
        
        # 分割数据集
        total_samples = len(all_patches)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # 创建子数据集
        train_patches = [all_patches[i] for i in train_indices]
        train_labels = [all_patch_labels[i] for i in train_indices]
        
        val_patches = [all_patches[i] for i in val_indices]  
        val_labels = [all_patch_labels[i] for i in val_indices]
        
        test_patches = [all_patches[i] for i in test_indices]
        test_labels = [all_patch_labels[i] for i in test_indices]
        
        # 创建PyTorch数据集
        train_dataset = TensorDataset(
            torch.stack(train_patches), 
            torch.stack(train_labels)
        )
        val_dataset = TensorDataset(
            torch.stack(val_patches), 
            torch.stack(val_labels)
        )
        test_dataset = TensorDataset(
            torch.stack(test_patches), 
            torch.stack(test_labels)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self):
        """创建数据加载器"""
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
```

### 2. 训练管理器

```python
class BrainConv3DTrainer:
    """3D脑卷积网络训练管理器"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device='cuda', learning_rate=1e-4):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader  
        self.test_loader = test_loader
        self.device = device
        
        # 优化器和损失函数
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略背景
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def train_epoch(self):
        """单个训练epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for patches, labels in pbar:
            patches, labels = patches.to(self.device), labels.to(self.device)
            
            # 标签调整：从1-102调整为0-101
            labels = labels - 1
            valid_mask = labels >= 0  # 过滤无效标签
            
            if valid_mask.sum() == 0:
                continue
                
            self.optimizer.zero_grad()
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                outputs = self.model(patches)
                loss = self.criterion(outputs[valid_mask], labels[valid_mask])
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs[valid_mask], 1)
            total += labels[valid_mask].size(0)
            correct += (predicted == labels[valid_mask]).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """验证阶段"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for patches, labels in tqdm(self.val_loader, desc='Validating'):
                patches, labels = patches.to(self.device), labels.to(self.device)
                labels = labels - 1
                valid_mask = labels >= 0
                
                if valid_mask.sum() == 0:
                    continue
                
                outputs = self.model(patches)
                loss = self.criterion(outputs[valid_mask], labels[valid_mask])
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs[valid_mask], 1)
                total += labels[valid_mask].size(0)
                correct += (predicted == labels[valid_mask]).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs=100, patience=15):
        """完整训练流程"""
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\\n=== Epoch {epoch+1}/{epochs} ===")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停和模型保存
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'history': self.history
                }, 'best_brain_conv3d_model.pth')
                
                print(f"新的最佳验证精度: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停于第 {epoch+1} 轮")
                    break
            
            self.scheduler.step()
        
        return self.history
```

## 使用示例代码

```python
def main():
    """完整的3D脑卷积网络训练示例"""
    
    # 1. 设置参数
    data_dir = "/path/to/3d/brain/data"
    patch_size = 3
    batch_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 创建数据加载器
    data_loader = BrainVolumeDataLoader(
        data_dir=data_dir,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=4
    )
    
    train_loader, val_loader, test_loader = data_loader.create_dataloaders()
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 3. 创建模型
    model = Conv3D_Voxel_Net(
        input_channels=351, 
        num_classes=102, 
        patch_size=patch_size
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 创建训练器
    trainer = BrainConv3DTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=1e-4
    )
    
    # 5. 开始训练
    history = trainer.train(epochs=100, patience=15)
    
    # 6. 可视化训练历史
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 精度曲线
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
```

## 参数量匹配架构设计（2025年1月更新）

### 新增：带可配置MLP头部的Conv2D模型

为了与1D全连接基线（~67M参数）进行公平对比，我们设计了参数量可配置的Conv2D架构，通过"轻量主干 + 大MLP头部"的策略达到目标参数量。

#### D) ImprovedConv2D_ParamMatched（参数对齐版）

```python
class ImprovedConv2D_ParamMatched(nn.Module):
    """带可配置MLP头部的Conv2D模型，用于参数量对齐"""
    
    def __init__(
        self,
        input_channels: int = 351,
        num_classes: int = 102,
        mid: int = 128,
        use_refine: bool = False,
        use_se: bool = True,
        kernel_size: int = 3,
        head_dims: Optional[Tuple[int, ...]] = None,
        mlp_hidden: int = 512,
        mlp_layers: int = 1,
        p_drop: float = 0.20,
        activation: str = 'silu'
    ):
        super().__init__()
        
        # === 主干网络（保持轻量） ===
        # 通道混合层
        self.mix = nn.Conv2d(input_channels, mid, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(8, mid)
        
        # 空间聚合层：3×3 -> 1×1
        self.agg = nn.Conv2d(mid, mid, kernel_size=kernel_size, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(8, mid)
        
        # 可选残差精炼
        if use_refine:
            self.refine = nn.Conv2d(mid, mid, kernel_size=1, bias=False)
            self.gn3 = nn.GroupNorm(8, mid)
        
        # SE注意力模块
        self.se = SE2D(mid) if use_se else nn.Identity()
        
        # === MLP头部（参数量主体） ===
        self._build_mlp_head(mid, num_classes, head_dims, 
                            mlp_hidden, mlp_layers, p_drop, activation)
    
    def _build_mlp_head(self, input_dim, output_dim, head_dims, 
                       mlp_hidden, mlp_layers, p_drop, activation):
        """构建可配置的MLP头部"""
        layers = []
        
        if head_dims is not None:
            # 使用精确指定的维度
            dims = [input_dim] + list(head_dims) + [output_dim]
        else:
            # 使用均匀宽度
            if mlp_layers == 1:
                dims = [input_dim, output_dim]  # 兼容原版
            else:
                dims = [input_dim] + [mlp_hidden] * mlp_layers + [output_dim]
        
        # 构建层序列
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # 中间层添加激活和dropout
            if i < len(dims) - 2:
                if activation == 'silu':
                    layers.append(nn.SiLU())
                else:
                    layers.append(nn.GELU())
                
                if p_drop > 0:
                    layers.append(nn.Dropout(p_drop))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        # 主干处理
        x = F.silu(self.gn1(self.mix(x)))
        x = F.silu(self.gn2(self.agg(x)))
        
        if hasattr(self, 'refine'):
            y = x
            x = F.silu(self.gn3(self.refine(x)))
            x = x + y
        
        x = self.se(x)
        x = x.flatten(1)  # (B, mid)
        
        # MLP头部
        return self.head(x)  # (B, num_classes)
```

### 参数量预设配置

提供三档预设，精确匹配不同规模的1D基线模型：

```python
def make_param_matched_model(
    target: str = "35M",
    input_channels: int = 351,
    num_classes: int = 102,
    kernel_size: int = 3,
    custom_head_dims: Optional[Tuple[int, ...]] = None,
    **kwargs
) -> ImprovedConv2D_ParamMatched:
    """工厂函数：创建参数量对齐的模型"""
    
    configs = {
        "35M": {
            "mid": 128,
            "head_dims": (4139, 4139, 4139)  # ~35M参数
        },
        "52M": {
            "mid": 768,
            "head_dims": (4608, 4608, 4608)  # ~52M参数
        },
        "69M": {
            "mid": 896,
            "head_dims": (4352, 4352, 4352, 4352)  # ~69M参数
        }
    }
    
    if target not in configs:
        raise ValueError(f"Unknown target: {target}")
    
    config = configs[target]
    if custom_head_dims is not None:
        config["head_dims"] = custom_head_dims
    
    return ImprovedConv2D_ParamMatched(
        input_channels=input_channels,
        num_classes=num_classes,
        kernel_size=kernel_size,
        **config,
        **kwargs
    )
```

### 参数量对比分析

| 配置 | 主干宽度(mid) | MLP头部结构 | 总参数量 | 误差 |
|------|--------------|------------|----------|------|
| **原始基线** | 128 | 单层(128→102) | ~58K | - |
| **35M匹配** | 128 | 3层(4139×3) | ~35M | <0.2% |
| **52M匹配** | 768 | 3层(4608×3) | ~52M | <0.2% |
| **69M匹配** | 896 | 4层(4352×4) | ~69M | <0.3% |

### 设计理念

1. **主干适中宽度**：保持Conv主干在合理范围（128-896），既能学习空间特征，又不过度膨胀
2. **大MLP头部**：通过深度MLP头部（3-4层）达到目标参数量，提供强大的特征变换能力
3. **渐进式配置**：
   - 35M：轻量主干(128) + 深MLP，适合资源受限场景
   - 52M：中等主干(768) + 深MLP，平衡性能与效率
   - 69M：宽主干(896) + 更深MLP，追求最高精度

### 使用示例

```python
# 方式1：使用预设配置
model_52m = make_param_matched_model(
    target="52M",
    input_channels=351,
    num_classes=102,
    kernel_size=3,
    use_se=True,
    use_refine=False
)

# 方式2：自定义配置
model_custom = ImprovedConv2D_ParamMatched(
    input_channels=351,
    num_classes=102,
    mid=512,
    head_dims=(4096, 4096, 2048),  # 自定义MLP结构
    p_drop=0.25,
    activation='gelu'
)

# 方式3：兼容模式（与原版行为一致）
model_compat = ImprovedConv2D_ParamMatched(
    input_channels=351,
    num_classes=102,
    mid=128,
    mlp_layers=1  # 单层头，参数量~58K
)
```

## 架构对比总结

| 架构 | 输入 | 参数量 | 学习内容 | 优势 | 适用场景 |
|------|------|--------|----------|------|----------|
| **1D FC** | 单体素(351,) | ~67M | 纯特征变换 | 简单直接 | 基线对比 |
| **2D Conv原版** | 平面patch(351,3,3) | ~58K | 2D空间+特征 | 极致轻量 | 快速推理 |
| **2D Conv-35M** | 平面patch(351,3,3) | ~35M | 2D空间+深度特征 | 参数效率高 | 中等算力 |
| **2D Conv-52M** | 平面patch(351,3,3) | ~52M | 2D空间+深度特征 | 性能平衡 | 推荐配置 |
| **2D Conv-69M** | 平面patch(351,3,3) | ~69M | 2D空间+深度特征 | 最高精度 | 充足算力 |
| **3D Conv** | 体积patch(351,3,3,3) | ~6M | 3D空间+特征 | 完整3D上下文 | 3D任务 |
| **多尺度3D** | 多尺度patches | ~15M | 多尺度空间关系 | 鲁棒性强 | 高精度需求 |

## 推荐配置

### 计算资源要求
- **GPU内存**: 至少8GB (推荐16GB+)
- **系统内存**: 至少32GB (推荐64GB+)
- **存储**: SSD，至少500GB可用空间

### 超参数建议
- **学习率**: 1e-4 (Adam优化器)
- **批次大小**: 256-512 (根据GPU内存调整)
- **Patch大小**: 3×3×3 (平衡性能和计算量)
- **权重衰减**: 1e-4
- **梯度裁剪**: max_norm=1.0

这个架构设计为您提供了从基础到高级的完整3D脑体素分类解决方案，您可以根据具体需求选择合适的网络结构和采样策略。

---

**文档版本**: 1.1  
**最后更新**: 2025年1月（新增参数匹配架构）  
**适用数据**: 384×336×256×351维多模态MRI数据  
**推荐起始点**: 
- 轻量快速：ImprovedConv2D_Baseline（~58K参数）
- 参数对齐：ImprovedConv2D_ParamMatched-52M（~52M参数）
- 3D任务：Conv3D_Voxel_Net + Brain3DPatchSet（~6M参数）