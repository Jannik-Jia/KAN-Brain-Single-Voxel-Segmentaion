#!/bin/bash

# =============================================================================
# 使用示例脚本
# Example Usage Script
# =============================================================================

echo "==============================================================================="
echo "              MRI体素分割训练系统 - 使用示例"
echo "==============================================================================="
echo ""

# =============================================================================
# 示例1: 快速测试不同模型（10 epochs）
# =============================================================================

echo "示例1: 快速测试不同模型"
echo "----------------------"
echo ""

echo "# 测试RegModel（原始模型）"
echo "MODEL_NAME=reg_model EPOCHS=10 ./run_training.sh"
echo ""

echo "# 测试ResNetMLP（残差网络）"
echo "MODEL_NAME=resnet_mlp EPOCHS=10 ./run_training.sh"
echo ""

echo "# 测试SimpleMLP（轻量级）"
echo "MODEL_NAME=simple_mlp EPOCHS=10 ./run_training.sh"
echo ""
echo ""

# =============================================================================
# 示例2: 完整训练（25 epochs）
# =============================================================================

echo "示例2: 完整训练"
echo "--------------"
echo ""

echo "# RegModel完整训练"
echo "MODEL_NAME=reg_model EPOCHS=25 BATCH_SIZE=8192 ./run_training.sh"
echo ""

echo "# ResNetMLP完整训练"
echo "MODEL_NAME=resnet_mlp EPOCHS=25 BATCH_SIZE=8192 ./run_training.sh"
echo ""

echo "# SimpleMLP完整训练"
echo "MODEL_NAME=simple_mlp EPOCHS=25 BATCH_SIZE=8192 ./run_training.sh"
echo ""
echo ""

# =============================================================================
# 示例3: 两种模式对比
# =============================================================================

echo "示例3: 两种模式对比（排除背景 vs 包含背景）"
echo "----------------------------------------"
echo ""

echo "# 排除背景模式（推荐）"
echo "MODEL_NAME=reg_model INCLUDE_BACKGROUND=false ./run_training.sh"
echo ""

echo "# 包含背景模式"
echo "MODEL_NAME=reg_model INCLUDE_BACKGROUND=true ./run_training.sh"
echo ""
echo ""

# =============================================================================
# 示例4: 使用Python脚本直接调用
# =============================================================================

echo "示例4: 使用Python脚本"
echo "-------------------"
echo ""

echo "# 基本使用"
echo "python train.py --model reg_model --epochs 25"
echo ""

echo "# 自定义参数"
echo "python train.py \\"
echo "    --model resnet_mlp \\"
echo "    --epochs 30 \\"
echo "    --batch_size 4096 \\"
echo "    --lr 0.0001 \\"
echo "    --hidden_dim 3072 \\"
echo "    --dropout_rate 0.4"
echo ""

echo "# 包含背景训练"
echo "python train.py --model reg_model --include_background --epochs 25"
echo ""
echo ""

# =============================================================================
# 示例5: 批量训练所有模型
# =============================================================================

echo "示例5: 批量训练所有模型"
echo "----------------------"
echo ""

echo "#!/bin/bash"
echo "# 保存为 train_all_models.sh"
echo ""
echo "# 定义模型列表"
echo "models=(reg_model resnet_mlp simple_mlp)"
echo ""
echo "# 循环训练每个模型"
echo "for model in \"\${models[@]}\"; do"
echo "    echo \"开始训练: \$model\""
echo "    MODEL_NAME=\$model EPOCHS=25 ./run_training.sh"
echo "    echo \"完成训练: \$model\""
echo "    echo \"\""
echo "done"
echo ""
echo "echo \"所有模型训练完成!\""
echo ""
echo ""

# =============================================================================
# 示例6: 自定义数据路径
# =============================================================================

echo "示例6: 自定义数据路径"
echo "-------------------"
echo ""

echo "# 使用环境变量"
echo "ROOT_DIR=/path/to/your/data MODEL_NAME=reg_model ./run_training.sh"
echo ""

echo "# 使用Python脚本"
echo "python train.py --model reg_model --root_dir /path/to/your/data"
echo ""
echo ""

# =============================================================================
# 实用技巧
# =============================================================================

echo "==============================================================================="
echo "实用技巧"
echo "==============================================================================="
echo ""

echo "1. 查看可用模型："
echo "   python train.py --help"
echo ""

echo "2. 查看训练日志："
echo "   tail -f logs/training_*.log"
echo ""

echo "3. 监控GPU使用："
echo "   watch -n 1 nvidia-smi"
echo ""

echo "4. 查看结果文件："
echo "   ls -lh results/"
echo ""

echo "5. 对比不同模型的训练曲线："
echo "   ls results/training_history_*.png"
echo ""

echo "==============================================================================="
echo ""

# =============================================================================
# 实际执行示例（取消注释以运行）
# =============================================================================

echo "如果要实际运行，请取消下面命令的注释："
echo ""

# 示例：快速测试SimpleMLP（10 epochs）
# MODEL_NAME=simple_mlp EPOCHS=10 BATCH_SIZE=8192 ./run_training.sh

# 示例：完整训练RegModel（25 epochs）
# MODEL_NAME=reg_model EPOCHS=25 BATCH_SIZE=8192 ./run_training.sh

echo "==============================================================================="
echo "更多详细信息，请查看 README.md"
echo "==============================================================================="
