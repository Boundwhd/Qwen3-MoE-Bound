#### 启动镜像

docker run -v ./workspace:/workspace --gpus all -it --rm qwen3-infer:latest

#### 切换到 python3.9
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1



