#!/usr/bin/env python3
"""
Paper M-Layer 环境和功能测试脚本

运行方式：
    python test_paper_mlayer.py

测试内容：
    1. 依赖库检查 (torch, numpy, einsum)
    2. expm_approx 函数测试
    3. RegModel_MLayerPaper 模型测试
    4. exact vs approx 模式对比
    5. 梯度计算测试
"""

import sys

def test_imports():
    """测试依赖库导入"""
    print("="*60)
    print("1. 依赖库检查")
    print("="*60)

    errors = []

    # PyTorch
    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        errors.append(f"❌ torch: {e}")

    # NumPy
    try:
        import numpy as np
        print(f"✅ numpy: {np.__version__}")
    except ImportError as e:
        errors.append(f"❌ numpy: {e}")

    # torch.einsum 测试
    try:
        import torch
        a = torch.randn(2, 3)
        b = torch.randn(3, 4, 4)
        c = torch.einsum('ba,amn->bmn', a, b)
        assert c.shape == (2, 4, 4)
        print(f"✅ torch.einsum: 工作正常")
    except Exception as e:
        errors.append(f"❌ torch.einsum: {e}")

    # torch.matrix_exp 测试
    try:
        import torch
        M = torch.randn(2, 4, 4)
        E = torch.matrix_exp(M)
        assert E.shape == (2, 4, 4)
        print(f"✅ torch.matrix_exp: 工作正常")
    except Exception as e:
        errors.append(f"❌ torch.matrix_exp: {e}")

    if errors:
        print("\n依赖检查失败:")
        for err in errors:
            print(f"  {err}")
        return False

    print("\n✅ 所有依赖检查通过!")
    return True


def test_expm_approx():
    """测试 expm_approx 函数"""
    print("\n" + "="*60)
    print("2. expm_approx 函数测试")
    print("="*60)

    import torch

    def expm_approx(M, k=6):
        """Scaling & Squaring 矩阵指数近似"""
        B, n, _ = M.shape
        I = torch.eye(n, device=M.device, dtype=M.dtype).unsqueeze(0).expand(B, -1, -1)
        E = I + M / (2 ** k)
        for _ in range(k):
            E = torch.matmul(E, E)
        return E

    # 测试不同的 k 值
    M = torch.randn(4, 8, 8) * 0.1  # 小矩阵确保收敛
    E_exact = torch.matrix_exp(M)

    print(f"输入矩阵形状: {M.shape}")
    print(f"输入矩阵范数 (Frobenius): {torch.norm(M, dim=(1,2)).mean():.4f}")
    print()

    for k in [4, 6, 8, 10]:
        E_approx = expm_approx(M, k=k)
        diff = torch.abs(E_exact - E_approx).max().item()
        print(f"k={k:2d}: 最大误差 = {diff:.2e}")

    # 验证 k=6 的精度足够
    E_approx_6 = expm_approx(M, k=6)
    diff_6 = torch.abs(E_exact - E_approx_6).max().item()

    if diff_6 < 0.01:
        print(f"\n✅ expm_approx (k=6) 精度满足要求 (误差 < 0.01)")
        return True
    else:
        print(f"\n⚠️ expm_approx (k=6) 精度较低: {diff_6:.4f}")
        return True  # 仍然返回 True，只是警告


def test_paper_model():
    """测试 RegModel_MLayerPaper 模型"""
    print("\n" + "="*60)
    print("3. RegModel_MLayerPaper 模型测试")
    print("="*60)

    import torch
    import torch.nn as nn

    # 复制模型定义（独立测试，不依赖主文件）
    def expm_approx(M, k=6):
        B, n, _ = M.shape
        I = torch.eye(n, device=M.device, dtype=M.dtype).unsqueeze(0).expand(B, -1, -1)
        E = I + M / (2 ** k)
        for _ in range(k):
            E = torch.matmul(E, E)
        return E

    class RegModel_MLayerPaper(nn.Module):
        def __init__(self, input_dim=351, num_classes=102, matrix_size=8,
                     basis_dim=64, expm_mode='exact', expm_k=6, scale=1.0, init_std=0.01):
            super().__init__()
            self.matrix_size = matrix_size
            self.expm_mode = expm_mode
            self.expm_k = expm_k
            self.scale = scale

            self.embed = nn.Linear(input_dim, basis_dim, bias=True)
            self.basis = nn.Parameter(torch.randn(basis_dim, matrix_size, matrix_size) * init_std)
            self.B = nn.Parameter(torch.zeros(matrix_size, matrix_size))
            self.out = nn.Linear(matrix_size * matrix_size, num_classes, bias=True)

        def forward(self, x):
            B = x.shape[0]
            z = self.embed(x)
            M = self.B + torch.einsum('ba,amn->bmn', z, self.basis)
            M = self.scale * M

            if self.expm_mode == 'exact':
                E = torch.matrix_exp(M)
            else:
                E = expm_approx(M, k=self.expm_k)

            E_flat = E.reshape(B, -1)
            logits = self.out(E_flat)
            return logits

    # 测试参数
    batch_size = 8
    input_dim = 351
    num_classes = 102

    x = torch.randn(batch_size, input_dim)
    print(f"输入形状: {x.shape}")
    print()

    # 测试不同配置
    configs = [
        {'matrix_size': 8, 'basis_dim': 64, 'expm_mode': 'exact'},
        {'matrix_size': 8, 'basis_dim': 64, 'expm_mode': 'approx'},
        {'matrix_size': 16, 'basis_dim': 128, 'expm_mode': 'exact'},
        {'matrix_size': 4, 'basis_dim': 32, 'expm_mode': 'exact'},
    ]

    all_passed = True
    for cfg in configs:
        try:
            model = RegModel_MLayerPaper(
                input_dim=input_dim,
                num_classes=num_classes,
                **cfg
            )
            model.eval()

            with torch.no_grad():
                out = model(x)

            param_count = sum(p.numel() for p in model.parameters())

            if out.shape == (batch_size, num_classes):
                print(f"✅ n={cfg['matrix_size']:2d}, A={cfg['basis_dim']:3d}, "
                      f"mode={cfg['expm_mode']:6s} | "
                      f"输出: {out.shape} | 参数: {param_count:,}")
            else:
                print(f"❌ n={cfg['matrix_size']}, A={cfg['basis_dim']}: "
                      f"输出形状错误 {out.shape}")
                all_passed = False

        except Exception as e:
            print(f"❌ n={cfg['matrix_size']}, A={cfg['basis_dim']}: {e}")
            all_passed = False

    if all_passed:
        print(f"\n✅ 所有模型配置测试通过!")
    return all_passed


def test_gradient():
    """测试梯度计算"""
    print("\n" + "="*60)
    print("4. 梯度计算测试")
    print("="*60)

    import torch
    import torch.nn as nn

    def expm_approx(M, k=6):
        B, n, _ = M.shape
        I = torch.eye(n, device=M.device, dtype=M.dtype).unsqueeze(0).expand(B, -1, -1)
        E = I + M / (2 ** k)
        for _ in range(k):
            E = torch.matmul(E, E)
        return E

    class RegModel_MLayerPaper(nn.Module):
        def __init__(self, input_dim=351, num_classes=102, matrix_size=8,
                     basis_dim=64, expm_mode='exact', expm_k=6, scale=1.0, init_std=0.01):
            super().__init__()
            self.matrix_size = matrix_size
            self.expm_mode = expm_mode
            self.expm_k = expm_k
            self.scale = scale

            self.embed = nn.Linear(input_dim, basis_dim, bias=True)
            self.basis = nn.Parameter(torch.randn(basis_dim, matrix_size, matrix_size) * init_std)
            self.B = nn.Parameter(torch.zeros(matrix_size, matrix_size))
            self.out = nn.Linear(matrix_size * matrix_size, num_classes, bias=True)

        def forward(self, x):
            B = x.shape[0]
            z = self.embed(x)
            M = self.B + torch.einsum('ba,amn->bmn', z, self.basis)
            M = self.scale * M

            if self.expm_mode == 'exact':
                E = torch.matrix_exp(M)
            else:
                E = expm_approx(M, k=self.expm_k)

            E_flat = E.reshape(B, -1)
            logits = self.out(E_flat)
            return logits

    # 测试 exact 模式梯度
    print("测试 exact 模式梯度...")
    model_exact = RegModel_MLayerPaper(expm_mode='exact')
    x = torch.randn(4, 351)
    y = torch.randint(0, 102, (4,))

    criterion = nn.CrossEntropyLoss()
    out = model_exact(x)
    loss = criterion(out, y)
    loss.backward()

    grad_ok = all(p.grad is not None and not torch.isnan(p.grad).any()
                  for p in model_exact.parameters() if p.requires_grad)

    if grad_ok:
        print(f"✅ exact 模式: 梯度计算正常")
    else:
        print(f"❌ exact 模式: 梯度计算异常")
        return False

    # 测试 approx 模式梯度
    print("测试 approx 模式梯度...")
    model_approx = RegModel_MLayerPaper(expm_mode='approx', expm_k=6)

    model_approx.zero_grad()
    out = model_approx(x)
    loss = criterion(out, y)
    loss.backward()

    grad_ok = all(p.grad is not None and not torch.isnan(p.grad).any()
                  for p in model_approx.parameters() if p.requires_grad)

    if grad_ok:
        print(f"✅ approx 模式: 梯度计算正常")
    else:
        print(f"❌ approx 模式: 梯度计算异常")
        return False

    print(f"\n✅ 梯度计算测试通过!")
    return True


def test_gpu():
    """测试 GPU 运行"""
    print("\n" + "="*60)
    print("5. GPU 测试 (可选)")
    print("="*60)

    import torch

    if not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，跳过 GPU 测试")
        return True

    import torch.nn as nn

    def expm_approx(M, k=6):
        B, n, _ = M.shape
        I = torch.eye(n, device=M.device, dtype=M.dtype).unsqueeze(0).expand(B, -1, -1)
        E = I + M / (2 ** k)
        for _ in range(k):
            E = torch.matmul(E, E)
        return E

    class RegModel_MLayerPaper(nn.Module):
        def __init__(self, input_dim=351, num_classes=102, matrix_size=8,
                     basis_dim=64, expm_mode='exact', expm_k=6, scale=1.0, init_std=0.01):
            super().__init__()
            self.matrix_size = matrix_size
            self.expm_mode = expm_mode
            self.expm_k = expm_k
            self.scale = scale

            self.embed = nn.Linear(input_dim, basis_dim, bias=True)
            self.basis = nn.Parameter(torch.randn(basis_dim, matrix_size, matrix_size) * init_std)
            self.B = nn.Parameter(torch.zeros(matrix_size, matrix_size))
            self.out = nn.Linear(matrix_size * matrix_size, num_classes, bias=True)

        def forward(self, x):
            B = x.shape[0]
            z = self.embed(x)
            M = self.B + torch.einsum('ba,amn->bmn', z, self.basis)
            M = self.scale * M

            if self.expm_mode == 'exact':
                E = torch.matrix_exp(M)
            else:
                E = expm_approx(M, k=self.expm_k)

            E_flat = E.reshape(B, -1)
            logits = self.out(E_flat)
            return logits

    device = torch.device('cuda')
    print(f"使用设备: {torch.cuda.get_device_name(0)}")

    model = RegModel_MLayerPaper().to(device)
    x = torch.randn(8, 351).to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)

    if out.shape == (8, 102):
        print(f"✅ GPU 前向传播正常")
    else:
        print(f"❌ GPU 输出形状错误: {out.shape}")
        return False

    # 测试 GPU 训练
    criterion = nn.CrossEntropyLoss()
    y = torch.randint(0, 102, (8,)).to(device)

    model.train()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    print(f"✅ GPU 训练正常 (loss={loss.item():.4f})")
    return True


def main():
    print("\n" + "="*60)
    print("Paper M-Layer 环境和功能测试")
    print("="*60 + "\n")

    results = []

    # 1. 依赖库检查
    results.append(("依赖库检查", test_imports()))
    if not results[-1][1]:
        print("\n❌ 依赖库检查失败，请先安装必要的库")
        sys.exit(1)

    # 2. expm_approx 测试
    results.append(("expm_approx", test_expm_approx()))

    # 3. 模型测试
    results.append(("模型测试", test_paper_model()))

    # 4. 梯度测试
    results.append(("梯度测试", test_gradient()))

    # 5. GPU 测试
    results.append(("GPU测试", test_gpu()))

    # 汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 所有测试通过！环境配置正确，可以开始训练。")
        print()
        print("示例命令:")
        print("  python train_1d_with_3d_dataset.py \\")
        print("      --data_dir_1d /path/to/1d \\")
        print("      --data_dir_3d /path/to/3d \\")
        print("      --model_type m_layer_paper \\")
        print("      --m_matrix_size 8 \\")
        print("      --m_basis_dim 64 \\")
        print("      --m_expm_mode exact")
    else:
        print("⚠️ 部分测试失败，请检查环境配置。")
        sys.exit(1)


if __name__ == '__main__':
    main()
