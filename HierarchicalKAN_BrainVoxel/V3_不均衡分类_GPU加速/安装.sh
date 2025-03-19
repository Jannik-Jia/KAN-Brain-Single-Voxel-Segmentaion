# 创建新的conda环境并安装基本依赖
conda create -n fastkan_gpu python=3.9 -y

# 激活环境


source /opt/conda/etc/profile.d/conda.sh
conda activate fastkan_gpu

# 安装CUDA工具包（根据你的GPU选择合适的CUDA版本）
conda install -c nvidia cuda-toolkit=11.8 -y

# 安装RAPIDS库（适用于GPU加速的数据科学库）
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=23.10 cudf=23.10 python=3.9 cudatoolkit=11.8 -y

# 安装ThunderSVM（GPU加速的SVM）
pip install thundersvm

# 安装常规数据科学库
conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn tqdm joblib -y

# 安装降维和可视化库
conda install -c conda-forge umap-learn -y
pip install openTSNE

# 安装PyTorch（可选，用于GPU加速的MLP等模型）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装其他工具
conda install -c conda-forge jupyterlab ipywidgets h5py -y

# 安装必要的特定包
git clone https://github.com/ZiyaoLi/fast-kan
cd fast-kan
pip install .



