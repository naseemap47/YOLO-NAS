FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git
RUN git clone https://github.com/naseemap47/YOLO-NAS.git home
WORKDIR /home
RUN pip install super-gradients==3.1.1
RUN pip install triton==2.0.0