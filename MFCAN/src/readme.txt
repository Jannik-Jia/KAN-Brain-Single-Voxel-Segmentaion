# 检查配置文件中的路径
cat configs/data_config.json

# 进行预处理
python scripts/preprocess_data.py --config configs/data_config.json

# 训练基线模型
python scripts/train_baseline.py --config configs/baseline_config.json