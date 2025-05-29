FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    libgomp1 \
    wget \
    curl \
    vim \
    iputils-ping \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

RUN python3 --version && pip3 --version

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install transformers accelerate numpy pandas

WORKDIR /workspace

CMD ["bash"]
