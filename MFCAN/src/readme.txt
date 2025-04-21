# 检查配置文件中的路径
cat configs/data_config.json

# 进行预处理
python scripts/preprocess_data.py --config configs/data_config.json

# 训练基线模型
python scripts/train_baseline.py --config configs/baseline_config.json



# 阶段一：数据分析
echo "Starting stage1_data_analysis.py..."
python scripts/stage1_data_analysis.py --config configs/data_config.json

# 阶段二：特征工程
echo "Starting stage2_feature_engineering.py..."
python scripts/stage2_feature_engineering.py --config configs/base_config.json

# 阶段二：模型训练
echo "Starting stage2_baseline_training.py..."
./run_stage2_baseline_training.sh --model_type deep_mlp --feature_type selected --feature_selection rf