import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import sys
from models.encoders.diffusion_encoder import DiffusionEncoder
from models.encoders.qti_encoder import QTIEncoder
from models.encoders.cest_encoder import CESTEncoder
from models.attention.class_conditioned_attention import ClassConditionedAttention
from models.fusion.cross_modal_fusion import CrossModalFusion
from models.classifiers.classification_head import ClassificationHead
from utils.logging_utils import Logger
    

class MFCAN(nn.Module):
    """
    多模态特征融合与类别适应网络(Multi-modal Feature fusion and Class-Adaptive Network)
    整合所有组件为一个端到端的网络，带有类别自适应机制
    """
    
    def __init__(self, config, pretrained_encoders=None):
        """
        初始化MFCAN架构
        
        参数:
            config: 配置字典，包含各组件的参数
            pretrained_encoders: 可选的预训练编码器字典 {'diffusion': path, 'qti': path, 'cest': path}
        """
        super(MFCAN, self).__init__()
        self.config = config
        


        # 使用已有的Logger类创建日志记录器
        log_manager = Logger("MFCAN", log_dir="logs/model")
        self.logger = log_manager.get_logger()
       

        # 初始化编码器
        self.diffusion_encoder = DiffusionEncoder(
            input_dim=config['diffusion_encoder']['input_dim'],
            hidden_dim=config['diffusion_encoder']['hidden_dim'],
            output_dim=config['diffusion_encoder']['output_dim'],
            dropout=config['diffusion_encoder']['dropout']
        )
        
        self.qti_encoder = QTIEncoder(
            input_dim=config['qti_encoder']['input_dim'],
            hidden_dims=config['qti_encoder']['hidden_dims'],
            output_dim=config['qti_encoder']['output_dim'],
            dropout=config['qti_encoder']['dropout']
        )
        
        self.cest_encoder = CESTEncoder(
            input_dim=config['cest_encoder']['input_dim'],
            hidden_dim=config['cest_encoder']['hidden_dim'],
            output_dim=config['cest_encoder']['output_dim'],
            dropout=config['cest_encoder']['dropout']
        )
        
        # 如果提供了预训练编码器路径，则加载权重
        if pretrained_encoders:
            self._load_pretrained_encoders(pretrained_encoders)
        
        # 类别条件注意力
        self.attention = ClassConditionedAttention(
            feature_dim=config['attention']['feature_dim'],
            num_classes=config['attention']['num_classes']
        )
        
        # 交叉模态融合
        self.fusion = CrossModalFusion(
            modal_dims=config['fusion']['modal_dims'],
            fusion_dim=config['main_classifier']['input_dim']
        )
        
        # 主分类器
        self.main_classifier = ClassificationHead(
            input_dim=config['main_classifier']['input_dim'],
            hidden_dim=config['main_classifier']['hidden_dim'],
            num_classes=config['main_classifier']['num_classes'],
            dropout=config.get('training', {}).get('dropout', 0.4)
        )
        
        # 辅助分类器（每个模态一个）
        self.auxiliary_classifiers = nn.ModuleDict({
            'diffusion': ClassificationHead(
                input_dim=config['diffusion_encoder']['output_dim'],
                hidden_dim=config['diffusion_encoder']['output_dim'] * 2,
                num_classes=config['main_classifier']['num_classes'],
                dropout=config.get('training', {}).get('dropout', 0.4)
            ),
            'qti': ClassificationHead(
                input_dim=config['qti_encoder']['output_dim'],
                hidden_dim=config['qti_encoder']['output_dim'] * 2,
                num_classes=config['main_classifier']['num_classes'],
                dropout=config.get('training', {}).get('dropout', 0.4)
            ),
            'cest': ClassificationHead(
                input_dim=config['cest_encoder']['output_dim'],
                hidden_dim=config['cest_encoder']['output_dim'] * 2,
                num_classes=config['main_classifier']['num_classes'],
                dropout=config.get('training', {}).get('dropout', 0.4)
            )
        })
        
        # 保存最近的注意力权重用于分析
        self.recent_attention_weights = None
    
    def _load_pretrained_encoders(self, pretrained_encoders):
        """
        加载预训练编码器权重
        
        参数:
            pretrained_encoders: 预训练编码器路径字典
        """
        # 加载diffusion编码器
        if 'diffusion' in pretrained_encoders and os.path.exists(pretrained_encoders['diffusion']):
            self._load_encoder(self.diffusion_encoder, pretrained_encoders['diffusion'], 'diffusion')
        
        # 加载qti编码器
        if 'qti' in pretrained_encoders and os.path.exists(pretrained_encoders['qti']):
            self._load_encoder(self.qti_encoder, pretrained_encoders['qti'], 'qti')
        
        # 加载cest编码器
        if 'cest' in pretrained_encoders and os.path.exists(pretrained_encoders['cest']):
            self._load_encoder(self.cest_encoder, pretrained_encoders['cest'], 'cest')
    
    def _load_encoder(self, encoder, path, name):
        """
        加载单个编码器的权重
        
        参数:
            encoder: 编码器模型
            path: 权重文件路径
            name: 编码器名称
        """
        # 获取logger，如果没有则使用默认logger
        logger = getattr(self, 'logger', logging.getLogger(__name__))
        
        # 尝试直接加载权重
        try:
            encoder.load_state_dict(torch.load(path))
            logger.info(f"成功加载{name}编码器权重: {path}")
            return
        except Exception as e:
            logger.warning(f"直接加载{name}编码器权重失败，尝试其他方法: {e}")
        
        # 尝试加载检查点
        try:
            checkpoint_path = path + ".ckpt"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                encoder.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"成功从检查点加载{name}编码器权重: {checkpoint_path}")
                return
        except Exception as e:
            logger.warning(f"从检查点加载{name}编码器权重失败: {e}")
        
        # 尝试兼容加载
        try:
            # 加载配置文件以获取更多信息
            config_path = path.replace('.pth', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    encoder_config = json.load(f)
                logger.info(f"加载了{name}编码器配置：{encoder_config}")
            
            # 尝试通过完整模型提取权重
            full_model_path = path + ".full"
            if os.path.exists(full_model_path):
                full_model = torch.load(full_model_path)
                encoder.load_state_dict(full_model.state_dict())
                logger.info(f"成功从完整模型加载{name}编码器权重: {full_model_path}")
                return
        except Exception as e:
            logger.warning(f"兼容方式加载{name}编码器权重失败: {e}")
        
        logger.warning(f"警告: 未能加载{name}编码器预训练权重，将使用随机初始化")

    
    def forward(self, x, class_priors=None, training_stage='all'):
        """
        前向传播
        
        参数:
            x: 包含三个模态特征的字典 {'diffusion': tensor, 'qti': tensor, 'cest': tensor}
            class_priors: 可选的类别先验信息 [batch_size]
            training_stage: 训练阶段 'encoders_only', 'fusion_only', 'all'
            
        返回:
            result: 包含主输出和辅助输出的字典
        """
        # 通过各编码器获取特征表示
        features = {}
        
        # 根据训练阶段设置梯度状态
        if training_stage == 'fusion_only':
            self.diffusion_encoder.eval()  # 添加这行
            self.qti_encoder.eval()        # 添加这行
            self.cest_encoder.eval()       # 添加这行
            with torch.no_grad():  # 冻结编码器
                features = {
                    'diffusion': self.diffusion_encoder(x['diffusion'], return_features=True),
                    'qti': self.qti_encoder(x['qti'], return_features=True),
                    'cest': self.cest_encoder(x['cest'], return_features=True)
                }
            # 恢复训练模式
            self.diffusion_encoder.train()  # 添加这行
            self.qti_encoder.train()        # 添加这行
            self.cest_encoder.train()       # 添加这行 
        else:
            features = {
                'diffusion': self.diffusion_encoder(x['diffusion'], return_features=True),
                'qti': self.qti_encoder(x['qti'], return_features=True),
                'cest': self.cest_encoder(x['cest'], return_features=True)
            }
        
        # 计算各辅助分类器的输出
        aux_outputs = {}
        if training_stage != 'fusion_only':
            aux_outputs = {
                'diffusion': self.auxiliary_classifiers['diffusion'](features['diffusion']),
                'qti': self.auxiliary_classifiers['qti'](features['qti']),
                'cest': self.auxiliary_classifiers['cest'](features['cest'])
            }
        
        # 如果只训练编码器，则不需要执行后续步骤
        if training_stage == 'encoders_only':
            return {
                'aux_outputs': aux_outputs
            }
        
        # 应用类别条件注意力
        attended_features, attention_weights = self.attention(features, class_priors)
        self.recent_attention_weights = attention_weights  # 保存用于分析
        
        # 交叉模态融合
        fused_features = self.fusion(features)
        
        # 主分类器
        main_output = self.main_classifier(fused_features)
        
        # 返回结果
        result = {
            'main_output': main_output,
            'aux_outputs': aux_outputs,
            'attention_weights': attention_weights
        }
        
        return result
    
    def get_attention_weights(self):
        """获取当前注意力权重分布"""
        return self.recent_attention_weights
    
    def freeze_encoders(self):
        """冻结所有编码器参数"""
        logger = getattr(self, 'logger', logging.getLogger(__name__))
        for param in self.diffusion_encoder.parameters():
            param.requires_grad = False
        for param in self.qti_encoder.parameters():
            param.requires_grad = False
        for param in self.cest_encoder.parameters():
            param.requires_grad = False
        logger.info("已冻结所有编码器参数")

    def unfreeze_encoders(self):
        """解冻所有编码器参数"""
        logger = getattr(self, 'logger', logging.getLogger(__name__))
        for param in self.diffusion_encoder.parameters():
            param.requires_grad = True
        for param in self.qti_encoder.parameters():
            param.requires_grad = True
        for param in self.cest_encoder.parameters():
            param.requires_grad = True
        logger.info("已解冻所有编码器参数")
    
    def get_parameter_groups(self):
        """
        获取参数分组，用于设置不同的学习率
        
        返回:
            parameter_groups: 参数分组列表
        """
        # 编码器参数（较小学习率）
        encoder_params = list(self.diffusion_encoder.parameters()) + \
                        list(self.qti_encoder.parameters()) + \
                        list(self.cest_encoder.parameters())
        
        # 融合和分类器参数（较大学习率）
        fusion_params = list(self.attention.parameters()) + \
                        list(self.fusion.parameters()) + \
                        list(self.main_classifier.parameters())
        
        # 辅助分类器参数
        aux_params = []
        for classifier in self.auxiliary_classifiers.values():
            aux_params.extend(list(classifier.parameters()))
        
        return [
            {'params': encoder_params, 'name': 'encoders'},
            {'params': fusion_params, 'name': 'fusion'},
            {'params': aux_params, 'name': 'auxiliary'}
        ]