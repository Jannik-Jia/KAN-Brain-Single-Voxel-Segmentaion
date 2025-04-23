import os
import json
import torch
import logging
import time
import traceback

class ModelIO:
    """模型保存和加载工具类"""
    
    def __init__(self, logger=None, config=None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # 从配置文件中获取模型IO选项，若未指定则使用默认值
        model_io_config = self.config.get('model_io', {})
        self.save_full_model = model_io_config.get('save_full_model', True)
        self.save_checkpoint = model_io_config.get('save_checkpoint', True)
        self.save_weights_only = model_io_config.get('save_weights_only', True)
        self.version_compatible = model_io_config.get('version_compatible', True)
    
    def save_encoder(self, model, save_path, modality, config, metadata=None):
        """
        保存编码器模型，支持多种保存方式
        
        参数:
            model: 要保存的PyTorch模型
            save_path: 保存路径
            modality: 模态名称('diffusion', 'qti', 'cest')
            config: 配置字典
            metadata: 要保存的额外元数据
        
        返回:
            bool: 保存是否成功
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        success = True
        
        try:
            # 获取模型结构信息
            if modality == 'diffusion':
                input_dim = model.embedding[0].in_features if hasattr(model, 'embedding') and hasattr(model.embedding[0], 'in_features') else config.get('diffusion_dim', 15)
                hidden_dim = config.get('diffusion_hidden_dim', 64)
                output_dim = model.classifier.in_features if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features') else config.get('diffusion_output_dim', 32)
                dropout = config.get('diffusion_dropout', 0.1)
                hidden_dims = None
            elif modality == 'qti':
                input_dim = model.feature_extractor[0].in_features if hasattr(model, 'feature_extractor') and hasattr(model.feature_extractor[0], 'in_features') else config.get('qti_dim', 210)
                hidden_dims = config.get('qti_hidden_dims', [512, 256])
                output_dim = model.classifier.in_features if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features') else config.get('qti_output_dim', 128)
                dropout = config.get('qti_dropout', 0.3)
                hidden_dim = None
            elif modality == 'cest':
                input_dim = model.feature_extractor[0].in_features if hasattr(model, 'feature_extractor') and hasattr(model.feature_extractor[0], 'in_features') else config.get('cest_dim', 116)
                hidden_dim = config.get('cest_hidden_dim', 256)
                output_dim = model.classifier.in_features if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features') else config.get('cest_output_dim', 128)
                dropout = config.get('cest_dropout', 0.5)
                hidden_dims = None
            else:
                # 未知模态，使用默认值并记录警告
                self.logger.warning(f"未知模态: {modality}，使用默认参数")
                input_dim = None
                hidden_dim = None
                hidden_dims = None
                output_dim = None
                dropout = None
            
            # 创建模型结构配置
            model_config = {
                'modality': modality,
                'model_type': model.__class__.__name__,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'hidden_dims': hidden_dims,
                'output_dim': output_dim,
                'dropout': dropout,
                'num_classes': config.get('num_classes', 102),
                'saved_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 添加元数据
            if metadata:
                model_config.update(metadata)
            
            # 保存结构配置
            config_path = save_path.replace('.pth', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=4)
            self.logger.info(f"模型配置已保存到 {config_path}")
            
            # 使用初始化时设置的保存选项
            save_full_model = self.save_full_model
            save_checkpoint = self.save_checkpoint
            save_weights_only = self.save_weights_only

            # 1. 保存完整模型
            if save_full_model:
                try:
                    # 在PyTorch 2.6+中，使用safe_globals来确保兼容性
                    if self._is_torch_version_26_or_higher():
                        with torch.serialization.safe_globals(['numpy.core.multiarray.scalar']):
                            torch.save(model, save_path + ".full")
                    else:
                        torch.save(model, save_path + ".full")
                    self.logger.info(f"完整模型已保存到 {save_path}.full")
                except Exception as e:
                    self.logger.error(f"保存完整模型失败: {e}")
                    success = False
            
            # 2. 保存状态字典（只有权重）
            if save_weights_only:
                try:
                    torch.save(model.state_dict(), save_path)
                    self.logger.info(f"模型权重已保存到 {save_path}")
                except Exception as e:
                    self.logger.error(f"保存模型权重失败: {e}")
                    success = False
            
            # 3. 保存用于检查点恢复的完整状态
            if save_checkpoint:
                try:
                    checkpoint = {
                        'model_class': model.__class__.__name__,
                        'model_state_dict': model.state_dict(),
                        'config': model_config,
                    }
                    # 在PyTorch 2.6+中，使用safe_globals来确保兼容性
                    if self._is_torch_version_26_or_higher():
                        with torch.serialization.safe_globals(['numpy.core.multiarray.scalar']):
                            torch.save(checkpoint, save_path + ".ckpt")
                    else:
                        torch.save(checkpoint, save_path + ".ckpt")
                    self.logger.info(f"模型检查点已保存到 {save_path}.ckpt")
                except Exception as e:
                    self.logger.error(f"保存模型检查点失败: {e}")
                    success = False
            
            return success
        except Exception as e:
            self.logger.error(f"保存模型过程中发生错误: {e}")
            self.logger.error(traceback.format_exc())
            return False
            
    def _is_torch_version_26_or_higher(self):
        """检查当前PyTorch版本是否为2.6或更高"""
        try:
            major, minor = map(int, torch.__version__.split('.')[:2])
            return (major > 2) or (major == 2 and minor >= 6)
        except:
            # 如果无法解析版本，返回False
            return False
    
    def load_encoder(self, load_path, device='cuda'):
        """加载编码器模型"""
        # 尝试多种加载方式
        
        # 1. 优先尝试加载检查点文件
        if os.path.exists(load_path + ".ckpt"):
            try:
                # 先尝试标准方式加载
                checkpoint = torch.load(load_path + ".ckpt", map_location=device)
                
                # 从配置中获取模型类
                model_class = checkpoint['model_class']
                config = checkpoint['config']
                modality = config['modality']
                
                # 导入对应的模型类
                if modality == 'diffusion':
                    from models.encoders.diffusion_encoder import DiffusionEncoder as ModelClass
                elif modality == 'qti':
                    from models.encoders.qti_encoder import QTIEncoder as ModelClass
                elif modality == 'cest':
                    from models.encoders.cest_encoder import CESTEncoder as ModelClass
                else:
                    raise ValueError(f"不支持的模态: {modality}")
                
                # 创建模型实例
                if modality == 'qti':
                    model = ModelClass(
                        input_dim=config['input_dim'],
                        hidden_dims=config['hidden_dims'],
                        output_dim=config['output_dim'],
                        dropout=config['dropout']
                    )
                else:
                    model = ModelClass(
                        input_dim=config['input_dim'],
                        hidden_dim=config['hidden_dim'],
                        output_dim=config['output_dim'],
                        dropout=config['dropout']
                    )
                
                # 加载状态字典
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                self.logger.info(f"从检查点 {load_path}.ckpt 成功加载模型")
                return model, config
                
            except Exception as e:
                self.logger.warning(f"从检查点加载失败: {e}")
        
        # 2. 尝试加载完整模型
        if os.path.exists(load_path + ".full"):
            try:
                # 加载完整模型
                with torch.serialization.safe_globals(['numpy.core.multiarray.scalar']):
                    model = torch.load(load_path + ".full", map_location=device)
                self.logger.info(f"从 {load_path}.full 成功加载完整模型")
                
                # 加载配置
                config_path = load_path.replace('.pth', '_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                else:
                    config = {'modality': 'unknown'}
                
                return model, config
            except Exception as e:
                self.logger.warning(f"加载完整模型失败: {e}")

        # 3. 尝试只加载权重
        if os.path.exists(load_path):
            try:
                # 先加载配置，确定模型结构
                config_path = load_path.replace('.pth', '_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    modality = config['modality']
                    
                    # 创建对应的模型
                    if modality == 'diffusion':
                        from models.encoders.diffusion_encoder import DiffusionEncoder
                        model = DiffusionEncoder(
                            input_dim=config['input_dim'],
                            hidden_dim=config['hidden_dim'],
                            output_dim=config['output_dim'],
                            dropout=config['dropout']
                        )
                    elif modality == 'qti':
                        from models.encoders.qti_encoder import QTIEncoder
                        model = QTIEncoder(
                            input_dim=config['input_dim'],
                            hidden_dims=config['hidden_dims'],
                            output_dim=config['output_dim'],
                            dropout=config['dropout']
                        )
                    elif modality == 'cest':
                        from models.encoders.cest_encoder import CESTEncoder
                        model = CESTEncoder(
                            input_dim=config['input_dim'],
                            hidden_dim=config['hidden_dim'],
                            output_dim=config['output_dim'],
                            dropout=config['dropout']
                        )
                    else:
                        raise ValueError(f"不支持的模态: {modality}")
                    
                    # 加载权重
                    if self._is_torch_version_26_or_higher():
                        with torch.serialization.safe_globals(['numpy.core.multiarray.scalar']):
                            model.load_state_dict(torch.load(load_path, map_location=device))
                    else:
                        model.load_state_dict(torch.load(load_path, map_location=device))
                        
                    model = model.to(device)
                    self.logger.info(f"从 {load_path} 成功加载模型权重")
                    return model, config
                else:
                    self.logger.error(f"未找到配置文件 {config_path}")
            except Exception as e:
                self.logger.error(f"加载模型权重失败: {e}")        
        self.logger.error(f"所有加载方式均失败，无法加载模型 {load_path}")
        return None, None