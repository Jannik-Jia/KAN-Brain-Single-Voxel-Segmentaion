# src/bilingual_logger.py

import os
import logging
from datetime import datetime

class BilingualLogger:
    def __init__(self, log_dir="logs", log_name=None):
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志名称
        if log_name is None:
            log_name = f"pseudoinverse_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.logger = logging.getLogger("pseudoinverse_experiment")
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(os.path.join(log_dir, log_name), encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def info(self, msg_cn, msg_en=None):
        """记录信息级别的日志"""
        if msg_en is None:
            self.logger.info(f"{msg_cn}")
        else:
            self.logger.info(f"{msg_cn} | {msg_en}")
    
    def warning(self, msg_cn, msg_en=None):
        """记录警告级别的日志"""
        if msg_en is None:
            self.logger.warning(f"{msg_cn}")
        else:
            self.logger.warning(f"{msg_cn} | {msg_en}")
    
    def error(self, msg_cn, msg_en=None):
        """记录错误级别的日志"""
        if msg_en is None:
            self.logger.error(f"{msg_cn}")
        else:
            self.logger.error(f"{msg_cn} | {msg_en}")