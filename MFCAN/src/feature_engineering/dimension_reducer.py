import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, UMAP
import h5py
import json
import joblib
from datetime import datetime

class DimensionReducer:
    """降维与特征变换工具，用于应用PCA、UMAP等降维技术"""
    
    def __init__(self, config_path=None, output_dir=None, logger=None):
        """
        初始化降维器
        
        参数:
            config_path: 配置文件路径
            output_dir: 输出目录
            logger: 日志记录器
        """
        # 加载配置
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # 设置输出目录
        self.output_dir = output_dir or self.config.get('output_dir', 'results/dimensionality')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logger
        if logger:
            self.logger.info(f"降维器初始化完成，输出目录: {self.output_dir}")
            
        # 存储变换器
        self.transformers = {}
        
        # 存储各特征组的PCA方差保留百分比
        self.pca_variance = {}
        
        # 存储分析结果
        self.reduction_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def apply_pca(self, train_features, val_features=None, test_features=None, 
                feature_group="all", variance=0.95):
        """
        应用PCA降维
        
        参数:
            train_features: 训练集特征
            val_features: 验证集特征
            test_features: 测试集特征
            feature_group: 特征组名称
            variance: 要保留的方差比例
            
        返回:
            transformed_data: 降维后的数据字典
        """
        if self.logger:
            self.logger.info(f"对 {feature_group} 组应用PCA (保留方差: {variance})")
        
        # 确定需要保留的主成分数量
        n_components = min(train_features.shape[0], train_features.shape[1])
        
        try:
            # 创建PCA模型
            pca = PCA(n_components=n_components)
            
            # 拟合模型
            pca.fit(train_features)
            
            # 计算累积方差
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # 找到满足方差阈值的最小主成分数
            n_components_threshold = np.argmax(cumulative_variance >= variance) + 1
            
            # 确保至少保留一个主成分
            n_components_threshold = max(1, n_components_threshold)
            
            if self.logger:
                variance_percent = cumulative_variance[n_components_threshold-1] * 100
                self.logger.info(f"选择 {n_components_threshold} 个主成分，保留 {variance_percent:.2f}% 的方差")
            
            # 存储PCA方差保留信息
            self.pca_variance[feature_group] = cumulative_variance[n_components_threshold-1] * 100
            
            # 创建新的PCA模型，只使用选定数量的主成分
            pca_final = PCA(n_components=n_components_threshold)
            
            # 变换数据
            train_transformed = pca_final.fit_transform(train_features)
            
            # 存储变换器
            self.transformers[f"pca_{feature_group}"] = pca_final
            
            # 创建返回字典
            transformed_data = {'train': train_transformed}
            
            # 变换验证集
            if val_features is not None:
                val_transformed = pca_final.transform(val_features)
                transformed_data['val'] = val_transformed
            
            # 变换测试集
            if test_features is not None:
                test_transformed = pca_final.transform(test_features)
                transformed_data['test'] = test_transformed
            
            # 保存降维结果
            self.reduction_results[f"pca_{feature_group}"] = {
                'method': 'pca',
                'original_dims': train_features.shape[1],
                'reduced_dims': n_components_threshold,
                'variance_explained': cumulative_variance[n_components_threshold-1],
                'timestamp': self.timestamp
            }
            
            # 可视化降维结果
            self._visualize_pca_results(pca_final, feature_group)
            
            return transformed_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"PCA降维失败: {e}")
            # 返回原始特征
            transformed_data = {'train': train_features}
            if val_features is not None:
                transformed_data['val'] = val_features
            if test_features is not None:
                transformed_data['test'] = test_features
            return transformed_data
    
    def apply_umap(self, features, labels=None, feature_group="all", n_components=2, 
                 n_neighbors=15, min_dist=0.1):
        """
        应用UMAP降维
        
        参数:
            features: 特征矩阵
            labels: 类别标签
            feature_group: 特征组名称
            n_components: 降维后的维度
            n_neighbors: 邻居数量
            min_dist: 最小距离
            
        返回:
            umap_data: UMAP降维结果
        """
        # 确保安装了umap-learn库
        try:
            from umap import UMAP
        except ImportError:
            if self.logger:
                self.logger.warning("未安装umap-learn库，请使用'pip install umap-learn'安装")
            return None
            
        if self.logger:
            self.logger.info(f"对 {feature_group} 组应用UMAP (维度: {n_components})")
        
        try:
            # 创建UMAP模型
            umap_model = UMAP(n_components=n_components, 
                              n_neighbors=n_neighbors, 
                              min_dist=min_dist,
                              random_state=42)
            
            # 如果提供了标签，用于监督降维
            if labels is not None:
                umap_result = umap_model.fit_transform(features, labels)
            else:
                umap_result = umap_model.fit_transform(features)
            
            # 存储变换器
            self.transformers[f"umap_{feature_group}"] = umap_model
            
            # 保存降维结果
            self.reduction_results[f"umap_{feature_group}"] = {
                'method': 'umap',
                'original_dims': features.shape[1],
                'reduced_dims': n_components,
                'timestamp': self.timestamp
            }
            
            # 可视化降维结果
            if n_components == 2:
                self._visualize_2d_embedding(umap_result, labels, "UMAP", feature_group)
            elif n_components == 3:
                self._visualize_3d_embedding(umap_result, labels, "UMAP", feature_group)
            
            return {'embedding': umap_result}
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"UMAP降维失败: {e}")
            return None
    
    def apply_custom_transformation(self, features, transform_type, feature_group="all"):
        """
        应用自定义特征变换
        
        参数:
            features: 特征矩阵
            transform_type: 变换类型，如'polynomial', 'interaction'等
            feature_group: 特征组名称
            
        返回:
            transformed_data: 变换后的数据
        """
        if self.logger:
            self.logger.info(f"对 {feature_group} 组应用自定义变换 (类型: {transform_type})")
        
        # 在这里实现自定义变换逻辑，根据transform_type选择不同的变换方法
        # 例如多项式特征、特征交互等
        
        # 暂时返回原始特征
        if self.logger:
            self.logger.warning(f"自定义变换 '{transform_type}' 尚未实现")
        return {'transformed': features}
    
    def _visualize_pca_results(self, pca_model, feature_group):
        """可视化PCA降维结果"""
        if self.logger:
            self.logger.info(f"可视化 {feature_group} 组的PCA结果")
        
        # 创建输出子目录
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            # 绘制解释方差比例
            plt.figure(figsize=(12, 8))
            
            explained_variance = pca_model.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            plt.subplot(2, 1, 1)
            plt.bar(range(1, len(explained_variance) + 1), explained_variance)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title(f'Explained Variance Ratio of {feature_group} Group')

            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title(f'Cumulative Explained Variance of {feature_group} Group')

            plt.grid(True, alpha=0.3)
            
            # 添加参考线
            for threshold in [0.8, 0.9, 0.95]:
                # 找到第一个超过阈值的索引
                idx = np.argmax(cumulative_variance >= threshold)
                n_components = idx + 1
                plt.axhline(y=threshold, color='g', linestyle='--', alpha=0.5)
                plt.axvline(x=n_components, color='g', linestyle='--', alpha=0.5)
                plt.text(n_components, threshold, f'  {n_components} components\n  {threshold*100:.0f}% variance', 
                       verticalalignment='center')

            
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{feature_group}_pca_variance.png")
            plt.savefig(save_path)
            plt.close()
            
            if self.logger:
                self.logger.info(f"PCA方差图已保存至 {save_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成PCA可视化失败: {e}")
    
    def _visualize_2d_embedding(self, embedding, labels, method, feature_group):
        """可视化二维降维结果"""
        if self.logger:
            self.logger.info(f"可视化 {feature_group} 组的{method}二维嵌入")
        
        # 创建输出子目录
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            plt.figure(figsize=(12, 10))
            
            # 如果有标签，用不同颜色绘制不同类别
            if labels is not None:
                unique_labels = np.unique(labels)
                
                # 如果类别太多，只使用前20个
                if len(unique_labels) > 20:
                    mask = np.isin(labels, unique_labels[:20])
                    plot_embedding = embedding[mask]
                    plot_labels = labels[mask]
                    unique_labels = unique_labels[:20]
                else:
                    plot_embedding = embedding
                    plot_labels = labels
                
                # 为每个类别使用不同颜色
                for i, label in enumerate(unique_labels):
                    mask = plot_labels == label
                    plt.scatter(plot_embedding[mask, 0], plot_embedding[mask, 1], 
                              label=f'Class {label}', alpha=0.6, s=50)
                
                plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')

            else:
                # 如果没有标签，使用单一颜色
                plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=50)
            
            plt.title(f'{feature_group} Group {method} 2D Embedding')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{feature_group}_{method.lower()}_2d.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f"{method}二维嵌入图已保存至 {save_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成{method}二维嵌入可视化失败: {e}")
    
    def _visualize_3d_embedding(self, embedding, labels, method, feature_group):
        """可视化三维降维结果"""
        if self.logger:
            self.logger.info(f"可视化 {feature_group} 组的{method}三维嵌入")
        
        # 创建输出子目录
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 如果有标签，用不同颜色绘制不同类别
            if labels is not None:
                unique_labels = np.unique(labels)
                
                # 如果类别太多，只使用前20个
                if len(unique_labels) > 20:
                    mask = np.isin(labels, unique_labels[:20])
                    plot_embedding = embedding[mask]
                    plot_labels = labels[mask]
                    unique_labels = unique_labels[:20]
                else:
                    plot_embedding = embedding
                    plot_labels = labels
                
                # 为每个类别使用不同颜色
                for i, label in enumerate(unique_labels):
                    mask = plot_labels == label
                    ax.scatter(plot_embedding[mask, 0], plot_embedding[mask, 1], plot_embedding[mask, 2],
                            label=f'Class {label}', alpha=0.6, s=50)
                
                ax.legend(title="类别", bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # 如果没有标签，使用单一颜色
                ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.6, s=50)

            ax.set_title(f'{feature_group} Group {method} 3D Embedding')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')

            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(vis_dir, f"{feature_group}_{method.lower()}_3d.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f"{method}三维嵌入图已保存至 {save_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成{method}三维嵌入可视化失败: {e}")
    
    def save_transformers(self, output_dir=None):
        """保存变换器模型供推理使用"""
        if not self.transformers:
            if self.logger:
                self.logger.warning("没有变换器可供保存")
            return
            
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'transformers')
            
        os.makedirs(output_dir, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"保存变换器到 {output_dir}")
            
        for name, transformer in self.transformers.items():
            try:
                # 保存变换器
                save_path = os.path.join(output_dir, f"{name}.joblib")
                joblib.dump(transformer, save_path)
                
                if self.logger:
                    self.logger.info(f"变换器 '{name}' 已保存至 {save_path}")
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"保存变换器 '{name}' 失败: {e}")
        
        # 保存变换器元数据
        metadata = {
            'timestamp': self.timestamp,
            'transformers': list(self.transformers.keys()),
            'pca_variance': self.pca_variance
        }
        
        try:
            metadata_path = os.path.join(output_dir, "transformers_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            if self.logger:
                self.logger.info(f"变换器元数据已保存至 {metadata_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存变换器元数据失败: {e}")
    
    def save_transformed_features(self, pca_features, umap_features, group_transformed_features, 
                               train_labels=None, val_labels=None, test_labels=None,
                               output_path=None):
        """
        保存变换后的特征
        
        参数:
            pca_features: PCA变换后的特征字典
            umap_features: UMAP变换后的特征字典
            group_transformed_features: 特征组变换后的特征字典
            train_labels: 训练集标签
            val_labels: 验证集标签
            test_labels: 测试集标签
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"transformed_features_{self.timestamp}.h5")
            
        if self.logger:
            self.logger.info(f"保存变换后的特征到 {output_path}")
            
        try:
            with h5py.File(output_path, 'w') as f:
                # 保存元数据
                f.attrs['timestamp'] = self.timestamp
                f.attrs['pca_variance'] = json.dumps(self.pca_variance)
                
                # 保存PCA特征
                if pca_features:
                    pca_group = f.create_group('pca')
                    for split, features in pca_features.items():
                        pca_group.create_dataset(split, data=features)
                        # 保存标签
                        if split == 'train' and train_labels is not None:
                            pca_group.create_dataset('train_labels', data=train_labels)
                        elif split == 'val' and val_labels is not None:
                            pca_group.create_dataset('val_labels', data=val_labels)
                        elif split == 'test' and test_labels is not None:
                            pca_group.create_dataset('test_labels', data=test_labels)
                
                # 保存UMAP特征
                if umap_features and 'embedding' in umap_features:
                    umap_group = f.create_group('umap')
                    umap_group.create_dataset('embedding', data=umap_features['embedding'])
                    if train_labels is not None:
                        umap_group.create_dataset('labels', data=train_labels)
                
                # 保存特征组变换后的特征
                if group_transformed_features:
                    for group_name, transformed in group_transformed_features.items():
                        group = f.create_group(group_name)
                        
                        # 保存PCA特征
                        if 'pca' in transformed:
                            pca_subgroup = group.create_group('pca')
                            for split, features in transformed['pca'].items():
                                pca_subgroup.create_dataset(split, data=features)
                        
                        # 保存UMAP特征
                        if 'umap' in transformed and 'embedding' in transformed['umap']:
                            umap_subgroup = group.create_group('umap')
                            umap_subgroup.create_dataset('embedding', data=transformed['umap']['embedding'])
            
            if self.logger:
                self.logger.info(f"变换后的特征已保存至 {output_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存变换后的特征失败: {e}")
    
    def generate_summary_report(self, output_path=None):
        """
        生成降维与特征变换的汇总报告
        
        参数:
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"dimensionality_reduction_summary_{self.timestamp}.md")
            
        if not self.reduction_results:
            if self.logger:
                self.logger.warning("没有可供汇总的降维结果")
            return
            
        try:
            with open(output_path, 'w') as f:
                f.write("# 降维与特征变换汇总报告\n\n")
                f.write(f"分析时间: {self.timestamp}\n\n")
                
                # PCA降维统计
                pca_results = {k: v for k, v in self.reduction_results.items() if 'pca' in k}
                
                if pca_results:
                    f.write("## PCA降维统计\n\n")
                    f.write("| 特征组 | 原始维度 | 降维后维度 | 降维比例 | 方差解释率 |\n")
                    f.write("|--------|----------|------------|----------|------------|\n")
                    
                    for key, result in pca_results.items():
                        group_name = key.replace('pca_', '')
                        original_dims = result['original_dims']
                        reduced_dims = result['reduced_dims']
                        reduction_ratio = (original_dims - reduced_dims) / original_dims * 100
                        variance_explained = result['variance_explained'] * 100
                        
                        f.write(f"| {group_name} | {original_dims} | {reduced_dims} | "
                               f"{reduction_ratio:.1f}% | {variance_explained:.1f}% |\n")
                    
                    f.write("\n")
                
                # UMAP降维统计
                umap_results = {k: v for k, v in self.reduction_results.items() if 'umap' in k}
                
                if umap_results:
                    f.write("## UMAP降维统计\n\n")
                    f.write("| 特征组 | 原始维度 | 降维后维度 | 降维比例 |\n")
                    f.write("|--------|----------|------------|----------|\n")
                    
                    for key, result in umap_results.items():
                        group_name = key.replace('umap_', '')
                        original_dims = result['original_dims']
                        reduced_dims = result['reduced_dims']
                        reduction_ratio = (original_dims - reduced_dims) / original_dims * 100
                        
                        f.write(f"| {group_name} | {original_dims} | {reduced_dims} | "
                               f"{reduction_ratio:.1f}% |\n")
                    
                    f.write("\n")
                
                # 降维建议
                f.write("## 降维建议\n\n")
                
                # 分析PCA结果
                if pca_results:
                    avg_reduction_ratio = np.mean([
                        (r['original_dims'] - r['reduced_dims']) / r['original_dims'] * 100 
                        for r in pca_results.values()
                    ])
                    
                    avg_variance_explained = np.mean([r['variance_explained'] * 100 for r in pca_results.values()])
                    
                    if avg_reduction_ratio > 70:
                        f.write("1. **PCA降维效率**: PCA能显著降低特征维度(平均减少{:.1f}%)，同时保留{:.1f}%的方差信息，建议在模型训练中使用PCA降维的特征。\n\n".format(
                            avg_reduction_ratio, avg_variance_explained
                        ))
                    elif avg_reduction_ratio > 30:
                        f.write("1. **PCA降维效率**: PCA能适度降低特征维度(平均减少{:.1f}%)，同时保留{:.1f}%的方差信息，在计算资源有限时可考虑使用。\n\n".format(
                            avg_reduction_ratio, avg_variance_explained
                        ))
                    else:
                        f.write("1. **PCA降维效率**: PCA降维效率不高(仅减少{:.1f}%)，特征间可能已经相对独立，建议直接使用原始特征或选择的特征子集。\n\n".format(
                            avg_reduction_ratio
                        ))
                
                # 特征组间比较
                if len(pca_results) > 1:
                    f.write("2. **特征组降维比较**:\n")
                    
                    for key, result in pca_results.items():
                        group_name = key.replace('pca_', '')
                        original_dims = result['original_dims']
                        reduced_dims = result['reduced_dims']
                        reduction_ratio = (original_dims - reduced_dims) / original_dims * 100
                        variance_explained = result['variance_explained'] * 100
                        
                        f.write(f"   - **{group_name}组**: 从{original_dims}维降至{reduced_dims}维(减少{reduction_ratio:.1f}%)，保留{variance_explained:.1f}%方差\n")
                    
                    f.write("\n")
                
                # 可视化说明
                f.write("## 降维可视化\n\n")
                
                f.write("可视化结果保存在visualizations目录中，包括：\n\n")
                
                if pca_results:
                    f.write("- **PCA方差解释图**: 展示了各主成分的方差解释比例和累积方差\n")
                
                if umap_results:
                    f.write("- **UMAP二维嵌入图**: 展示了使用UMAP将高维特征映射到二维空间的结果，有助于观察数据分布和类别分离情况\n")
                
                f.write("\n")
                
                # 模型训练建议
                f.write("## 模型训练建议\n\n")
                
                f.write("基于降维分析，在模型训练中可考虑以下策略：\n\n")
                
                f.write("1. **降维特征使用**: 建议比较以下特征集的模型性能：\n")
                f.write("   - 原始特征集或特征选择后的子集\n")
                f.write("   - PCA降维后的特征集\n")
                f.write("   - 不同特征组的组合特征\n\n")
                
                f.write("2. **降维参数调优**: 如果使用降维特征，可以调整以下参数：\n")
                f.write("   - PCA的方差保留比例(如90%、95%、99%)\n")
                f.write("   - 对不同特征组使用不同的降维策略\n\n")
                
                f.write("3. **特征融合方法**: 可以考虑以下特征融合策略：\n")
                f.write("   - 早期融合：先合并所有特征，再进行降维\n")
                f.write("   - 中期融合：对每个特征组单独降维，再合并降维后的特征\n")
                f.write("   - 晚期融合：对每个特征组单独建模，再合并模型预测结果\n")
            
            if self.logger:
                self.logger.info(f"降维与特征变换汇总报告已保存至 {output_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成降维与特征变换汇总报告失败: {e}")