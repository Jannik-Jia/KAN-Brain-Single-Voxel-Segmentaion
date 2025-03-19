# 创建一个目录来存放软链接（如果需要）
mkdir -p ~/lib

# 创建软链接
ln -s /opt/conda/pkgs/libcusparse-dev-11.7.5.86-0/lib/libcusparse.so ~/lib/libcusparse.so.9.0

# 设置库路径
export LD_LIBRARY_PATH=~/lib:$LD_LIBRARY_PATH

# 运行你的程序
python main.py