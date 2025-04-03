# test_gpu.py
try:
    import cupy as cp
    print(f"CuPy 已安装，版本: {cp.__version__}")
    print(f"CUDA 设备数量: {cp.cuda.runtime.getDeviceCount()}")
    print(f"CUDA 设备名称: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    
    # 测试简单的 GPU 计算
    a = cp.arange(10)
    b = cp.arange(10)
    c = a + b
    print(f"测试计算结果: {c}")
    print("GPU 可用且正常工作")
except ImportError:
    print("CuPy 未安装。请使用以下命令安装:")
    print("pip install cupy-cuda11x  # 替换为您的 CUDA 版本")
except Exception as e:
    print(f"错误: {str(e)}")
    print("CUDA 环境可能配置不正确")
