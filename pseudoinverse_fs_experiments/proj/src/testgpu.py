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
    
    print(f"正在初始化GPU，use_gpu参数值: {use_gpu}")
    
    if not use_gpu:
        print("已设置为使用CPU，跳过GPU初始化")
        USE_GPU = False
        xp = np
        return False
    
    try:
        print("尝试导入CuPy...")
        import cupy as cp
        
        # 限制GPU内存使用
        try:
            print("设置GPU内存分配器...")
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
            cp.cuda.memory.set_limit_ratio(memory_fraction)
            print(f"已设置GPU内存使用上限为{memory_fraction*100:.0f}%")
        except Exception as e:
            print(f"设置GPU内存限制时出错: {str(e)}")
        
        # 检查CUDA是否可用
        print("检查CUDA可用性...")
        if cp.cuda.is_available():
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
            
    except ImportError as e:
        print(f"导入CuPy时出错: {str(e)}")
        print("未安装CuPy，切换到CPU模式")
        print("要使用GPU加速，请安装CuPy: pip install cupy-cuda11x (根据CUDA版本选择)")
        USE_GPU = False
        xp = np
        return False
    except Exception as e:
        print(f"GPU初始化过程中发生其他错误: {str(e)}")
        USE_GPU = False
        xp = np
        return False
