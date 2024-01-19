FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
COPY . /home
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6
WORKDIR /home
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install pytorch-quantization==2.1.2 --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install super-gradients==3.1.3