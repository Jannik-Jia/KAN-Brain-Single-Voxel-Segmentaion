# 创建libcusparse.so.9.0的符号链接（如果libcusparse.so.11.7存在）
sudo ln -s /usr/local/cuda/lib64/libcusparse.so.11.7 /usr/local/cuda/lib64/libcusparse.so.9.0

# 或者设置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
