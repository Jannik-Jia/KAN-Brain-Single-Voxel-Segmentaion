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



# 增强gpu_utils.py中的函数，减少传输次数，添加批处理支持

def to_gpu(x, force_copy=False):
    """
    将数组转移到GPU，添加缓存机制避免重复传输
    
    参数:
        x: 输入数组
        force_copy: 是否强制复制数据
        
    返回:
        数组: GPU上的数组
    """
    if USE_GPU:
        import cupy as cp
        if isinstance(x, np.ndarray):
            if hasattr(x, '_gpu_cached') and not force_copy:
                return x._gpu_cached
            gpu_array = cp.asarray(x)
            # 尝试在原始数组上缓存GPU版本的引用
            try:
                x._gpu_cached = gpu_array
            except:
                pass
            return gpu_array
        elif isinstance(x, cp.ndarray):
            return x.copy() if force_copy else x
    return x

def to_cpu(x, force_copy=False):
    """
    将数组转移到CPU，添加缓存避免重复传输
    
    参数:
        x: 输入数组
        force_copy: 是否强制复制数据
        
    返回:
        数组: CPU上的NumPy数组
    """
    if USE_GPU:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            if hasattr(x, '_cpu_cached') and not force_copy:
                return x._cpu_cached
            cpu_array = cp.asnumpy(x)
            # 尝试在原始数组上缓存CPU版本的引用
            try:
                x._cpu_cached = cpu_array
            except:
                pass
            return cpu_array
        elif isinstance(x, np.ndarray):
            return x.copy() if force_copy else x
    return x

def ensure_numpy(x):
    """
    确保输出是NumPy数组，修复隐式转换问题
    
    参数:
        x: 输入数组
        
    返回:
        numpy.ndarray: NumPy数组
    """
    if USE_GPU:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)  # 使用显式转换
    return to_cpu(x)  # 使用我们增强的to_cpu函数

def batch_operation(operation, *args, device='gpu'):
    """
    在指定设备上批量执行操作，减少数据传输
    
    参数:
        operation: 要执行的函数
        args: 传递给operation的参数
        device: 'gpu'或'cpu'，指定在哪个设备上执行
        
    返回:
        操作结果
    """
    if device == 'gpu' and USE_GPU:
        # 确保所有参数在GPU上
        gpu_args = [to_gpu(arg) for arg in args]
        # 在GPU上执行操作
        result = operation(*gpu_args)
        # 不立即转回CPU
        return result
    else:
        # 确保所有参数在CPU上
        cpu_args = [to_cpu(arg) for arg in args]
        # 在CPU上执行操作
        return operation(*cpu_args)