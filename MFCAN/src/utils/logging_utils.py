import os
import logging
from datetime import datetime
import sys
import json

class Logger:
    """统一的日志记录器"""
    
    def __init__(self, name, log_dir='logs', console_level=logging.INFO, file_level=logging.DEBUG):
        """
        初始化日志记录器
        
        参数:
            name: 日志名称
            log_dir: 日志目录
            console_level: 控制台日志级别
            file_level: 文件日志级别
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = name
        self.log_dir = os.path.join(log_dir, name.lower().replace(' ', '_'))
        self.console_level = console_level
        self.file_level = file_level
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 获取日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # 设置为最低级别，让handler决定
        
        # 防止重复添加handler
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # 设置日志记录器
        self._setup_logger()
        
        self.logger.info(f"Logger '{name}' 初始化完成, 日志存储在 {self.log_dir}")
    
    def _setup_logger(self):
        """配置日志记录器"""
        # 日志文件路径
        log_file = os.path.join(self.log_dir, f"{self.name.lower().replace(' ', '_')}_{self.timestamp}.log")
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.file_level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 保存handler引用
        self.file_handler = file_handler
        self.console_handler = console_handler
    
    def get_logger(self):
        """获取日志记录器"""
        return self.logger
    
    def update_file_level(self, level):
        """更新文件日志级别"""
        self.file_handler.setLevel(level)
        self.logger.info(f"文件日志级别更新为 {level}")
    
    def update_console_level(self, level):
        """更新控制台日志级别"""
        self.console_handler.setLevel(level)
        self.logger.info(f"控制台日志级别更新为 {level}")
    
    def log_config(self, config, title="Configuration"):
        """记录配置信息"""
        self.logger.info(f"===== {title} =====")
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    self.logger.info(f"{key}:")
                    for k, v in value.items():
                        self.logger.info(f"  {k}: {v}")
                else:
                    self.logger.info(f"{key}: {value}")
        else:
            self.logger.info(str(config))
        self.logger.info("=" * (len(title) + 12))
    
    def log_metrics(self, metrics, step=None, prefix=""):
        """记录指标信息"""
        msg = f"{prefix} Metrics"
        if step is not None:
            msg += f" (Step {step})"
        self.logger.info(f"===== {msg} =====")
        
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, dict):
                    self.logger.info(f"{key}:")
                    for k, v in value.items():
                        self.logger.info(f"  {k}: {v}")
                else:
                    self.logger.info(f"{key}: {value}")
        else:
            self.logger.info(str(metrics))
        self.logger.info("=" * (len(msg) + 12))
    
    def log_experiment_start(self, experiment_name, description=None):
        """记录实验开始"""
        self.logger.info("=" * 80)
        self.logger.info(f"开始实验: {experiment_name}")
        if description:
            self.logger.info(f"描述: {description}")
        self.logger.info("时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("=" * 80)
        
    def log_experiment_end(self, experiment_name, results=None):
        """记录实验结束"""
        self.logger.info("=" * 80)
        self.logger.info(f"结束实验: {experiment_name}")
        self.logger.info("时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        if results:
            self.logger.info("实验结果:")
            if isinstance(results, dict):
                for key, value in results.items():
                    self.logger.info(f"  {key}: {value}")
            else:
                self.logger.info(f"  {results}")
                
        self.logger.info("=" * 80)
    
    def log_error(self, error, context=None):
        """记录错误信息"""
        self.logger.error(f"错误: {error}")
        if context:
            self.logger.error(f"上下文: {context}")
            
    def save_to_json(self, data, filename):
        """将数据保存为JSON文件"""
        try:
            output_path = os.path.join(self.log_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"数据已保存至 {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"保存JSON数据失败: {e}")
            return None