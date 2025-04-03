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
        
        # 限制GPU内存使用
        try:
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
            cp.cuda.memory.set_limit_ratio(memory_fraction)
            print(f"已设置GPU内存使用上限为{memory_fraction*100:.0f}%")
        except Exception as e:
            print(f"设置GPU内存限制时出错: {str(e)}")
        
        # 检查CUDA是否可用
        if cp.cuda.is_available():
            print(f"GPU初始化成功: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
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
        warnings.warn("要使用GPU加速，请安装CuPy: pip install cupy-cuda11x (根据CUDA版本选择)")
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