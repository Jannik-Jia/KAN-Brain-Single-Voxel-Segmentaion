# src/gpu_utils.py

import warnings
import numpy as np

# 全局变量，表示是否使用GPU
USE_GPU = False
xp = np

def init_gpu(use_gpu=True, memory_fraction=0.8):
    """
    初始化GPU支持
    
    参数:
        use_gpu: 是否使用GPU
        memory_fraction: GPU内存使用比例上限
        
    返回:
        bool: 是否成功启用GPU
    """
    global USE_GPU, xp
    
    if not use_gpu:
        print("已设置为使用CPU，跳过GPU初始化")
        USE_GPU = False
        xp = np
        return False
    
    try:
        import cupy as cp
        
        # 检查CUDA是否可用
        if cp.cuda.is_available():
            # 尝试设置内存池（不影响主要功能）
            try:
                cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
                # 注意：较新版本的CuPy可能没有set_limit_ratio方法
                # 我们尝试使用替代方法或跳过这一步
                try:
                    cp.cuda.memory.set_limit_ratio(memory_fraction)
                    print(f"已设置GPU内存使用上限为{memory_fraction*100:.0f}%")
                except AttributeError:
                    print(f"当前CuPy版本不支持set_limit_ratio方法，使用默认内存管理")
                    # 可能的替代方法（取决于CuPy版本）
                    if hasattr(cp.cuda, 'setMemoryFraction'):
                        cp.cuda.setMemoryFraction(memory_fraction)
                        print(f"使用setMemoryFraction设置内存比例为{memory_fraction}")
            except Exception as e:
                print(f"设置GPU内存管理时出错: {str(e)}，将使用默认内存管理")
                
            # 不管内存设置如何，我们仍然启用GPU
            device_name = cp.cuda.runtime.getDeviceProperties(0)['name']
            print(f"GPU初始化成功: {device_name}")
            USE_GPU = True
            xp = cp
            return True
        else:
            print("CUDA不可用，切换到CPU模式")
            USE_GPU = False
            xp = np
            return False
            
    except ImportError:
        print("未安装CuPy，切换到CPU模式")
        print("要使用GPU加速，请安装CuPy: pip install cupy-cuda11x (根据CUDA版本选择)")
        USE_GPU = False
        xp = np
        return False

def get_array_module(x):
    """
    根据输入数组返回对应的数组模块
    
    参数:
        x: 输入数组
        
    返回:
        模块: numpy或cupy
    """
    if USE_GPU:
        import cupy as cp
        return cp.get_array_module(x)
    return np

def to_gpu(x):
    """
    将数组转移到GPU
    
    参数:
        x: 输入数组
        
    返回:
        数组: GPU上的数组
    """
    if USE_GPU:
        import cupy as cp
        if isinstance(x, np.ndarray):
            return cp.asarray(x)
    return x

def to_cpu(x):
    """
    将数组转移到CPU
    
    参数:
        x: 输入数组
        
    返回:
        数组: CPU上的NumPy数组
    """
    if USE_GPU:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    return x

def ensure_numpy(x):
    """
    确保输出是NumPy数组
    
    参数:
        x: 输入数组
        
    返回:
        numpy.ndarray: NumPy数组
    """
    return to_cpu(x)