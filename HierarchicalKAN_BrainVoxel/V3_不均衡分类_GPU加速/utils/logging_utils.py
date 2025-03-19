#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志工具模块，提供统一的日志记录功能
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LOG_LEVEL, LOG_FORMAT, LOG_FILENAME, LOGS_DIR

# 确保日志目录存在
os.makedirs(LOGS_DIR, exist_ok=True)

# 配置日志格式
formatter = logging.Formatter(LOG_FORMAT)

def get_logger(name):
    """
    获取配置好的日志记录器
    
    参数:
        name: 日志记录器名称，通常是模块名
        
    返回:
        logger: 日志记录器对象
    """
    logger = logging.getLogger(name)
    
    # 设置日志级别
    level = getattr(logging, LOG_LEVEL)
    logger.setLevel(level)
    
    # 如果已经有处理器，则不再添加
    if logger.handlers:
        return logger
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_section(logger, section_name, level=logging.INFO):
    """
    记录一个新的日志部分，用于清晰区分不同的执行阶段
    
    参数:
        logger: 日志记录器对象
        section_name: 部分名称
        level: 日志级别
    """
    separator = "=" * 80
    logger.log(level, separator)
    logger.log(level, f" {section_name} ".center(80, "="))
    logger.log(level, separator)

def log_execution_time(logger, start_time, end_time=None, prefix="执行时间"):
    """
    记录执行时间
    
    参数:
        logger: 日志记录器对象
        start_time: 开始时间
        end_time: 结束时间，如果为None则使用当前时间
        prefix: 日志前缀
    """
    if end_time is None:
        end_time = datetime.now()
    
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        logger.info(f"{prefix}: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    elif minutes > 0:
        logger.info(f"{prefix}: {int(minutes)}分钟 {seconds:.2f}秒")
    else:
        logger.info(f"{prefix}: {seconds:.2f}秒")

# 测试
if __name__ == "__main__":
    logger = get_logger("test")
    
    log_section(logger, "测试日志部分")
    logger.debug("这是一条调试信息")
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")
    
    start_time = datetime.now()
    # 模拟一些处理
    import time
    time.sleep(1.5)
    log_execution_time(logger, start_time)
    
    print("日志工具测试完成")